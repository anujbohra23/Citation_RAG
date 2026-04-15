from pathlib import Path
from typing import List
import xml.etree.ElementTree as ET

from src.core.types import Document


def load_medquad_documents(root_dir: str) -> List[Document]:
    docs: List[Document] = []
    root = Path(root_dir)

    xml_files = list(root.rglob("*.xml"))

    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
            xml_root = tree.getroot()
        except Exception:
            continue

        for qa in xml_root.iter():
            children = list(qa)
            if not children:
                continue

            child_tags = {child.tag.lower() for child in children}
            if "question" not in child_tags or "answer" not in child_tags:
                continue

            question = ""
            answer = ""
            qtype = ""
            focus = ""

            for child in children:
                tag = child.tag.lower()
                text = (child.text or "").strip()

                if tag == "question":
                    question = text
                elif tag == "answer":
                    answer = text
                elif tag in {"qtype", "questiontype"}:
                    qtype = text
                elif tag == "focus":
                    focus = text

            if not question or not answer:
                continue

            collection = xml_path.parent.name
            doc_id = f"{collection}_{len(docs)}"

            docs.append(
                Document(
                    doc_id=doc_id,
                    title=question[:120],
                    text=answer,
                    source_path=str(xml_path),
                    metadata={
                        "question": question,
                        "question_type": qtype,
                        "focus": focus,
                        "collection": collection,
                        "dataset": "MedQuAD",
                    },
                )
            )

    return docs