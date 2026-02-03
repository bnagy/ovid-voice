import bs4  # type: ignore
from bs4 import BeautifulSoup
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import os


def _process_poem(poem: bs4.Tag) -> dict[str, str]:
    out = []
    for l in poem("div", "line"):
        if "ellipse" in l["class"]:
            continue
        for w in l("span", "word"):
            try:
                for s in w.children:
                    if "long" in s["class"]:
                        out.append("S")
                    elif "short" in s["class"]:
                        out.append("w")
                    elif "elided" in s["class"]:
                        continue
                    else:
                        raise ValueError(f"Unknown syllable class {s['class']}")
            except Exception as e:
                raise RuntimeError(
                    f"Error while parsing\n    Word: {w}\n    Line: {l}\n"
                ) from e
            out.append(".")
        # Metronome format lines end with wordbreak _and_ linebreak symbols .|
        # I now regret this decision, but it's too late to change
        out.append("|")

    metre = ""
    m = poem.find("div", "poem_meter")
    if m:
        metre = m.text
    else:
        metre = poem["data-metre"]

    h = {
        "author": poem["data-author"],
        "work": poem["data-work"],
        "meter": metre,
        "book": "",
        "number": "",
        "title": "",
        "lines": out.count("|"),
        "metronome": "".join(out),
    }
    if poem.has_attr("data-book"):
        h["book"] = poem["data-book"]
    if poem.has_attr("data-number"):
        h["number"] = poem["data-number"]

    # sometimes there's a data-title in the poem div. Sometimes we need to look
    # in the poem_header div. In the header it can be either poem_title or
    # work_title. As far as I know there's nowhere where data-title exists and
    # the poem_header is not filled in so that seems more reliable.
    pt = poem.find("div", "poem_title")
    if pt and len(pt.text) > 0:
        h["title"] = pt.text.title()
    wt = poem.find("div", "work_title")
    if not pt and wt and len(wt.text) > 0:
        h["title"] = wt.text.title()

    return h


def _extract_text(poem: bs4.Tag) -> dict[str, str]:
    out = []
    for l in poem("div", "line"):
        if "ellipse" in l["class"]:
            continue
        this_line = []
        for w in l("span", "word"):
            try:
                word = "".join([s.text for s in w.children])
                this_line.append(word)
            except Exception as e:
                raise RuntimeError(
                    f"Error while parsing\n    Word: {w}\n    Line: {l}\n"
                ) from e
        out.append(" ".join(this_line))

    metre = ""
    m = poem.find("div", "poem_meter")
    if m:
        metre = m.text
    else:
        metre = poem["data-metre"]

    h = {
        "author": poem["data-author"],
        "work": poem["data-work"],
        "meter": metre,
        "book": "",
        "number": "",
        "title": "",
        "lines": len(out),
        "text": "\n".join(out),
    }
    if poem.has_attr("data-book"):
        h["book"] = poem["data-book"]
    if poem.has_attr("data-number"):
        h["number"] = poem["data-number"]

    # sometimes there's a data-title in the poem div. Sometimes we need to look
    # in the poem_header div. In the header it can be either poem_title or
    # work_title. As far as I know there's nowhere where data-title exists and
    # the poem_header is not filled in so that seems more reliable.
    pt = poem.find("div", "poem_title")
    if pt and len(pt.text) > 0:
        h["title"] = pt.text.title()
    wt = poem.find("div", "work_title")
    if not pt and wt and len(wt.text) > 0:
        h["title"] = wt.text.title()

    return h


def extract_file(fn: str) -> pd.DataFrame:
    """
    Process one file in the hypotactic html format. Extract raw text for each poem. The file may
    contain multiple poems, sometimes with titles, sometimes with numbers.

    In:
        fn (str): name of the file to process

    Returns:
        pd.DataFrame: pandas DataFrame containing the results
    """
    with open(fn) as fh:
        poems = BeautifulSoup(fh, "lxml").find_all("div", "poem")
    df = pd.DataFrame([_extract_text(p) for p in poems])

    if df.number.any() and not df.number.is_unique:
        # if the number column has actual numbers in it, try to de-dup them
        s = df.number
        letters = np.array([""] + list("bcdef"))
        # the count is an index into the letter array, so first entry, 0=no suffix,
        # first dup gets b etc
        df.number = s + letters[s.groupby(s).cumcount()]
        if not df.number.is_unique:
            raise ValueError(
                "ERROR: tried to make numbers unique but too many duplicates!"
            )
    return df


def process_file(fn: str) -> pd.DataFrame:
    """
    Process one file in the hypotactic html format. Converts the scansion into
    the metronome format. The file may contain multiple poems, sometimes with
    titles, sometimes with numbers.

    In:
        fn (str): name of the file to process

    Returns:
        pd.DataFrame: pandas DataFrame containing the results
    """
    with open(fn) as fh:
        poems = BeautifulSoup(fh, "lxml").find_all("div", "poem")
    df = pd.DataFrame([_process_poem(p) for p in poems])

    if df.number.any() and not df.number.is_unique:
        # if the number column has actual numbers in it, try to de-dup them
        s = df.number
        letters = np.array([""] + list("bcdef"))
        # the count is an index into the letter array, so first entry, 0=no suffix,
        # first dup gets b etc
        df.number = s + letters[s.groupby(s).cumcount()]
        if not df.number.is_unique:
            raise ValueError(
                "ERROR: tried to make numbers unique but too many duplicates!"
            )
    return df


def process_directory(dn: str) -> pd.DataFrame:
    """
    Process one directory full of files in the hypotactic html format. Converts
    the scansion into the metronome format. The files may contain multiple
    poems, sometimes with titles, sometimes with numbers. This will attempt to
    process _all_ files in the given directory (not just *.html), but will not
    recurse into subdirectories.

    In:
        dn (str): name of the directory to process

    Returns:
        pd.DataFrame: pandas DataFrame containing the results
    """
    if not os.path.isdir(dn):
        raise ValueError(f"{dn} is not a directory.")

    files = [
        os.path.join(dn, f)
        for f in os.listdir(dn)
        if os.path.isfile(os.path.join(dn, f))
    ]
    dfs = []
    for f in files:
        try:
            dfs.append(process_file(f))
        except Exception as e:
            raise ValueError(f"Error processing {f}") from e

    return pd.concat(dfs).reset_index(drop=True)


def canonical_title(r: pd.Series) -> str:
    """
    Convenience method to try to determine a sensible title. Intended to be used like:
       ```
       col = df.apply(canonical_title, axis=1)
       df.insert(0, 'canonical_title', col)
       ```
    """
    if r["number"]:
        if r["book"]:
            return f"{r['work']} {r['book']}.{r['number']}"
        else:
            return f"{r['work']} {r['number']}"
    elif r["book"]:
        return f"{r['work']} {r['book']}"
    elif r["title"]:
        return str(r["title"])
    else:
        return f"{r['work']} {r.name}"
