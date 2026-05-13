"""SEC HTML filing loader using BeautifulSoup."""
from pathlib import Path
from bs4 import BeautifulSoup
from langchain_core.documents import Document


def load_sec_filing(filepath: str) -> Document:
    """Load a single SEC HTML filing and extract clean text.
    
    SEC filings have inline XBRL tags, tables, and noise.
    BeautifulSoup strips HTML and keeps substantive text.
    Metadata (ticker, source, filing_type) propagates to chunks.
    """
    filename = Path(filepath).name
    
    with open(filepath, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, "lxml")
    
    for element in soup(["script", "style"]):
        element.decompose()
    
    text = soup.get_text(separator="\n", strip=True)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    clean_text = "\n".join(lines)
    
    ticker = filename.split("_")[0]
    
    return Document(
        page_content=clean_text,
        metadata={
            "source": filename,
            "ticker": ticker,
            "filing_type": "10-K"
        }
    )


def load_all_filings(data_dir: str = "data") -> list[Document]:
    """Load all .htm filings from a directory."""
    docs = []
    for filepath in sorted(Path(data_dir).glob("*.htm")):
        docs.append(load_sec_filing(str(filepath)))
    return docs
