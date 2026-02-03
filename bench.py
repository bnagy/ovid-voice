import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Callable

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, Normalizer, FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, make_pipeline

unigrams = make_pipeline(
    CountVectorizer(
        lowercase=False,
        analyzer="word",
        tokenizer=lambda x: x.split(" "),
        token_pattern=None,
    ),
    FunctionTransformer(lambda x: np.array(x.todense()), accept_sparse=True),
)

bigrams = make_pipeline(
    CountVectorizer(
        lowercase=False,
        analyzer="word",
        tokenizer=lambda x: x.split(" "),
        token_pattern=None,
        ngram_range=(2, 2),
    ),
    FunctionTransformer(lambda x: np.array(x.todense()), accept_sparse=True),
)

unibi = make_pipeline(
    CountVectorizer(
        lowercase=False,
        analyzer="word",
        tokenizer=lambda x: x.split(" "),
        token_pattern=None,
        ngram_range=(1, 2),
    ),
    FunctionTransformer(lambda x: np.array(x.todense()), accept_sparse=True),
)

bitri = make_pipeline(
    CountVectorizer(
        lowercase=False,
        analyzer="word",
        tokenizer=lambda x: x.split(" "),
        token_pattern=None,
        ngram_range=(2, 3),
    ),
    FunctionTransformer(lambda x: np.array(x.todense()), accept_sparse=True),
)

text = make_pipeline(
    CountVectorizer(
        lowercase=True,
        analyzer="char",
        ngram_range=(2, 5),
    ),
    FunctionTransformer(lambda x: np.array(x.todense()), accept_sparse=True),
)


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and preprocesses a DataFrame containing parsed Latin sentences.

    This function performs the following operations:
        - Filters out sentences that are too short or too long based on the number of tokens.
        - Cleans problematic characters from the 'feats' column.
        - Removes certain inflectional features from 'feats' and stores the result in 'feats_no_infl'.
        - Creates two modified versions of the 'tree' column:
            - 'head_tree': removes signed distance information from dependency labels.
            - 'sign_tree': replaces signed distances with direction indicators (+/-).

    Args:
        df (pd.DataFrame): Input DataFrame with columns including 'text', 'pos', 'feats', and 'tree'.

    Returns:
        pd.DataFrame: Cleaned and preprocessed DataFrame.
    """
    # make a copy so we're not working on a view
    df = df.copy()
    # keep only reasonable length sentences
    limit = int(df.text.str.split(" ").str.len().quantile(0.975))
    df = df[(df.pos.str.count(" ") > 4) & (df.pos.str.count(" ") < limit)]
    # remove characters from the features that cause problems
    df["feats"] = df.feats.replace(r"\[", "(", regex=True)
    df["feats"] = df.feats.replace(r"\]", ")", regex=True)
    df["rawtree"] = df["tree"].replace(r"_.+?\)", "", regex=True)
    df["feats_no_infl"] = df.feats.replace("InflClass.*? ", "", regex=True)
    df["feats_no_infl"] = df["feats_no_infl"].replace("Aspect=Imp ", "", regex=True)
    # We create two modified versions of the tree. The ones from the client contain the signed
    # distance between the element and its head in parens like: nmod_adv(3). The variable head_tree
    # does not contain the paren section at all, the variable sign_tree just contains the direction (before or
    # after the head in the sentence) as (+) or (-)
    df.insert(6, "head_tree", df.tree.replace(r"\(.+?\)", "", regex=True))

    # Create sign_tree, as above. First mask out the root_ISAT() feature which just records the
    # (positive) distance of the sentence root from the start of the sentence.
    df["signed_tree"] = df["tree"].str.replace(
        r"(\w*ISAT\w*)\((\d+)\)", r"\1(TEMP\2)", regex=True
    )
    # Step 2: Replace all other number patterns
    df["signed_tree"] = df["signed_tree"].str.replace(
        r"(\w+)\((\d+)\)", r"\1(+)", regex=True
    )
    df["signed_tree"] = df["signed_tree"].str.replace(
        r"(\w+)\((-\d+)\)", r"\1(-)", regex=True
    )

    # Step 3: Restore ISAT patterns
    df["signed_tree"] = df["signed_tree"].str.replace(
        r"(\w*ISAT\w*)\(TEMP(\d+)\)", r"\1(\2)", regex=True
    )
    return df


def vec_scale_group(
    df: pd.DataFrame, cv: Pipeline, feat: str, squish: int = 1, norm: bool = True
) -> pd.DataFrame:
    """
    Vectorizes and scales grouped text data from a DataFrame.
    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data to be processed.
        cv (dict): A dictionary containing a CountVectorizer instance under the key "countvectorizer".
        feat (str): The name of the column in the DataFrame to be vectorized.
        squish (int, optional): The grouping factor to aggregate rows. Defaults to 1 (no grouping).
        norm (bool, optional): Whether to normalize the resulting vectors. Defaults to True.
    Returns:
        pd.DataFrame: A DataFrame containing the vectorized and optionally normalized features,
                      with columns corresponding to the feature names from the CountVectorizer.
    """

    # vectorise the text using the supplied vectorizer
    column = df.groupby(df.index // squish)[feat].agg(" ".join)
    ary = np.array(cv.transform(column))
    if norm:
        ary = Normalizer(norm="l1").fit_transform(ary)

    return pd.DataFrame(ary, columns=cv["countvectorizer"].get_feature_names_out())  # type: ignore


def df_scale_group(
    df,
    vec,
    feat,
    groups=["author", "work"],
    start_col: int = 4,
    squish=1,
    norm=True,
    scale=True,
    fit=True,
):
    """
    Scales and normalizes grouped data in a DataFrame while preserving group-specific information.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the data to be processed.
    vec : sklearn.feature_extraction.text.CountVectorizer or similar
        A vectorizer object used to fit and transform the `feat` column.
    feat : str
        The name of the column in `df` to be vectorized and scaled.
    squish : int, optional, default=1
        The factor by which to group rows for text aggregation.
    norm : bool, optional, default=True
        Whether to normalize the vectorized features.
    scale : bool, optional, default=True
        Whether to apply standard scaling to the resulting features.

    Returns:
    --------
    pandas.DataFrame
        A new DataFrame with scaled and normalized features, along with group-specific metadata.

    Notes:
    ------
    - The function groups the input DataFrame by "author" and "work" columns.
    - For each group, it vectorizes the specified feature column, optionally normalizes it,
        and aggregates text data based on the `squish` parameter.
    - The resulting DataFrame includes the original group metadata ("author", "work", "text")
        and the scaled feature columns.
    """
    coll = []
    # make sure we have the full vocab (number of columns) in all of the group dataframes
    if fit:
        vec.fit(df[feat])
    for g, gdf in df.groupby(groups):
        gdf.reset_index(inplace=True)
        this = pd.DataFrame(vec_scale_group(gdf, vec, feat, squish=squish, norm=norm))
        text = gdf.groupby(gdf.index // squish)["text"].agg(" ".join)
        # take the first entry in the columns up to start_col
        meta = gdf.iloc[::squish, :start_col].reset_index(drop=True)
        this = pd.concat([meta, this], axis=1)
        this.insert(0, "text", text.values)
        coll.append(this)
    df = pd.concat(coll, axis=0).reset_index(drop=True)
    if scale:
        # we added the text column at the front, so scale from start_col + 1
        X = pd.DataFrame(StandardScaler().fit_transform(df.iloc[:, start_col + 1 :]))
        X.columns = df.columns[start_col + 1 :]
        df = pd.concat([df.iloc[:, : start_col + 1], X], axis=1)
    return df


def signed_headvec(
    df: pd.DataFrame,
    groups: list[str],
    start_col: int = 4,
    squish=1,
    scale=True,
    norm=True,
) -> pd.DataFrame:
    """
    Generate a combined DataFrame by scaling and normalizing specific groups of features.

    This function processes a DataFrame by applying scaling and normalization to
    different feature groups ('rawtree', 'signed_tree', and 'feats') and then combines
    the processed groups into a single DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing raw data to be processed.
        squish (int, optional): A parameter to control the squishing factor during scaling. Defaults to 1.
        scale (bool, optional): Whether to apply scaling to the feature groups. Defaults to True.
        norm (bool, optional): Whether to apply normalization to the feature groups. Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing the combined and processed feature groups.
    """
    posunibi = df_scale_group(
        df,
        unibi,
        "pos",
        groups=groups,
        start_col=start_col,
        squish=squish,
        scale=scale,
        norm=norm,
    )
    treeunibi = df_scale_group(
        df,
        unigrams,
        "signed_tree",
        groups=groups,
        start_col=start_col,
        squish=squish,
        scale=scale,
        norm=norm,
    ).iloc[:, start_col + 1 :]
    feats = df_scale_group(
        df,
        unigrams,
        "feats",
        groups=groups,
        start_col=start_col,
        squish=squish,
        scale=scale,
        norm=norm,
    ).iloc[:, start_col + 1 :]
    combined = pd.concat([posunibi, treeunibi, feats], axis=1)
    return combined


def vectorize_squish(
    df: pd.DataFrame,
    fn: Callable[
        [pd.DataFrame, list[str], int, int, bool, bool], pd.DataFrame
    ] = signed_headvec,
    groups=["author", "work"],
    start_col: int = 4,
    squish: int = 1,
    scale: bool = True,
    norm: bool = True,
) -> pd.DataFrame:
    """
    Apply a vectorization function to a DataFrame with optional squishing, scaling, and normalization.

    Args:
        df (pd.DataFrame): The input DataFrame to be processed.
        fn (callable, optional): The vectorization function to apply. Defaults to `basevec`.
        squish (int, optional): A parameter to control the squishing effect. Defaults to 1.
        scale (bool, optional): Whether to scale the data. Defaults to True.
        norm (bool, optional): Whether to normalize the data. Defaults to True.

    Returns:
        pd.DataFrame: The transformed DataFrame after applying the vectorization function.
    """
    return fn(df, groups, start_col, squish, scale, norm)


def avg_classification_report(
    X, y, model, labels, n_splits=5, test_size=0.2, random_state=42
):
    """
    Generates an average classification report across multiple stratified splits.

    Args:
        X (array-like): Feature data.
        y (array-like): Target labels.
        model: scikit-learn classifier.
        n_splits (int): Number of splits for cross-validation.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        dict: Average classification report.
    """
    sss = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=random_state
    )
    report_dict = defaultdict(list)

    cms = []
    for train_index, test_index in sss.split(X, y):
        X = X.copy()
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        assert isinstance(
            report, dict
        ), "Expected classification report to be a dictionary"
        cm = confusion_matrix(y_pred, y_test)
        cms.append(cm)
        for k, v in report.items():
            if k in ["macro avg", "weighted avg"]:
                for kk, vv in v.items():
                    report_dict[f"{k}_{kk}"].append(vv)
            elif k == "accuracy":
                report_dict["accuracy"].append(v)
            elif k.isdigit():
                label = labels[int(k)]
                if report_dict[label]:
                    for kk, vv in v.items():
                        report_dict[label][kk].append(vv)
                else:
                    report_dict[label] = defaultdict(list)  # type: ignore
                    for kk, vv in v.items():
                        report_dict[label][kk].append(vv)

    average_report = {}
    for k, v in report_dict.items():
        if isinstance(v, list):
            average_report[k] = np.mean(v)
        elif isinstance(v, dict):
            for kk, vv in v.items():  # type: ignore
                if not average_report.get(k):
                    average_report[k] = {}
                average_report[k][kk] = np.mean(vv)
        else:
            # Handle unexpected types
            raise ValueError(f"Unexpected type for key {k}: {type(v)}")

    return average_report, cms


def print_avg_report(average_report: dict, nsplits: int, test_size: float):
    """
    Prints an average classification report in a formatted manner.
    Args:
        average_report (dict): A dictionary containing the average classification
            metrics. The keys are metric names, and the values can either be
            floats or nested dictionaries with detailed metrics.
        nsplits (int): The number of splits used in the evaluation.
        test_size (float): The proportion of the dataset used as the test set.
    Example:
        average_report = {
            "accuracy": 0.95,
            "precision": {"class_1": 0.90, "class_2": 0.92},
            "recall": {"class_1": 0.88, "class_2": 0.91}
        }
        print_avg_report(average_report, nsplits=5, test_size=0.2)
    Returns:
        None: This function prints the average classification report to the console.
    """
    print(
        f"Average Classification Report ({nsplits} splits with test size {test_size}):"
    )
    for k, v in average_report.items():
        if isinstance(v, dict):
            arr = []
            for x, y in v.items():
                if x == "support":
                    s = f"{x} {y:.0f}"
                else:
                    s = f"{x} {y:.1%}"
                arr.append(s)
            print(f"{k:25.25s} {'  '.join(arr)}")
        else:
            if "support" in k:
                s = f"{v:.0f}"
            else:
                s = f"{v:.1%}"
            print(f"{k:25.25s} {s}")


def bench_and_fit(
    df: pd.DataFrame,
    model,
    start_col: int = 4,
    y_col: str = "author",
    nsplits: int = 25,
    test_size: float = 0.1,
    quiet: bool = False,
):
    """
    Benchmarks and fits a classification model using cross-validation and displays results.
    Parameters:
        df (pd.DataFrame): The input DataFrame containing features and target variable.
                           The target variable is assumed to be in the 'author' column.
        model: The classification model to be trained and evaluated.
        nsplits (int, optional): The number of splits for cross-validation. Default is 25.
        test_size (float, optional): The proportion of the dataset to include in the test split. Default is 0.1.
        quiet (bool, optional): If True, suppresses output. Default is False.
    Returns:
        model: The trained classification model.
        average_report (dict): The average classification report across the cross-validation splits.
    Side Effects:
        - Prints the average classification report if `quiet` is False.
        - Displays a confusion matrix plot if `quiet` is False.
    Example:
        model, report = bench_and_fit(df, model, nsplits=10, test_size=0.2, quiet=False)
    Notes:
        - Prints the average classification report for the cross-validation.
        - Displays a confusion matrix plot with normalized values.
    """

    df = df.copy()
    X = df.iloc[:, start_col:]
    y, y_uniques = df[y_col].factorize()

    average_report, cms = avg_classification_report(
        X=X, y=y, labels=y_uniques, model=model, n_splits=nsplits, test_size=test_size
    )
    if not quiet:
        print_avg_report(average_report, nsplits, test_size)
        # average n confusion matrices by i,j index
        ConfusionMatrixDisplay(
            confusion_matrix=Normalizer(norm="l1").fit_transform(np.mean(cms, axis=0)),
            display_labels=list(y_uniques),
        ).plot(xticks_rotation="vertical", colorbar=False)
        plt.show()

    return (model, X, y), average_report
