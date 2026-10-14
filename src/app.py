"""Local RAG chat with original PDF pages and linked parser images."""
import hashlib
from html import unescape
import io
import json
import os
from pathlib import Path
import re
from threading import RLock

import pypdfium2 as pdfium
import streamlit as st

from llm_client import stream_chat
from rag import PROFILES, ROOT, build_context, retrieve

PDF = ROOT / "documents/B2B Dashboard Manual_v3.pdf"
PROFILE_LABELS = {"vl": "PaddleOCR-VL", "structure": "PP-StructureV3", "existing": "기존 Structure · 원본 대조 보정"}


@st.cache_resource
def resource_lock():
    # PDFium is not thread-safe; serialize rendering and local model requests.
    return RLock()


@st.cache_data(max_entries=3)
def read_corpus(profile, modified):
    path = PROFILES[profile][0]
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@st.cache_data(max_entries=1)
def pdf_info(modified):
    data = PDF.read_bytes()
    with resource_lock(), pdfium.PdfDocument(str(PDF)) as pdf:
        count = len(pdf)
    return data, count, hashlib.sha256(data).hexdigest()


@st.cache_data(max_entries=24)
def page_preview(page_number, modified):
    with resource_lock(), pdfium.PdfDocument(str(PDF)) as pdf:
        page = pdf[page_number - 1]
        try:
            bitmap = page.render(scale=1400 / page.get_width())
            try:
                image = bitmap.to_pil()
                try:
                    buffer = io.BytesIO()
                    image.save(buffer, format="PNG")
                    return buffer.getvalue()
                finally:
                    image.close()
            finally:
                bitmap.close()
        finally:
            page.close()


@st.cache_data(max_entries=128)
def image_bytes(relative, expected_hash, modified):
    path = (ROOT / relative).resolve()
    path.relative_to(ROOT.resolve())
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError("추출 이미지가 분석 당시 기록과 달라졌습니다.")
    return data


def first_page(record):
    included = [doc["page"] for doc in record["documents"] if not doc["skipped"]]
    cited = [int(page) for page in re.findall(r"\[p\.(\d+)\]", record["answer"])]
    return next((page for page in cited if page in included), included[0] if included else 1)


def select_source(profile, record_index, page):
    st.session_state[f"record_{profile}"] = record_index
    st.session_state[f"page_{profile}"] = page


def select_record(profile):
    history = st.session_state["histories"][profile]
    index = st.session_state[f"record_{profile}"]
    st.session_state[f"page_{profile}"] = first_page(history[index])


def display_text(content):
    # Make extracted text readable without executing parser-produced HTML.
    text = re.sub(r"</t[dh]\s*>", " | ", content, flags=re.I)
    text = re.sub(r"</tr\s*>", "\n\n", text, flags=re.I)
    return unescape(re.sub(r"<[^>]+>", "", text)).strip()


def main():
    st.set_page_config(page_title="매뉴얼 RAG", page_icon="📖", layout="wide")
    st.markdown("""<style>
      .block-container {max-width: 1480px; padding-top: 2rem;}
      h1 {letter-spacing: -0.04em;}
      [data-testid="stChatMessage"] {border: 1px solid #e3e8f1; border-radius: 14px;}
    </style>""", unsafe_allow_html=True)
    profiles = list(PROFILE_LABELS)
    default_profile = os.environ.get("B2B_RAG_PROFILE", "vl")
    default_device = os.environ.get("B2B_RAG_DEVICE", "gpu")
    with st.sidebar:
        st.title("📖 매뉴얼 RAG")
        profile = st.selectbox("문서 분석 방식", profiles,
                               index=profiles.index(default_profile) if default_profile in profiles else 0,
                               format_func=PROFILE_LABELS.get)
        device = st.radio("실행 장치", ["gpu", "cpu"], horizontal=True,
                          index=0 if default_device == "gpu" else 1, format_func=str.upper)
        top_k = st.select_slider("검색 근거 수", options=[1, 3, 5], value=3)
        st.divider()
        st.caption("답변은 추출된 텍스트·표를 사용합니다. 이미지는 원문을 직접 확인할 수 있도록 보여드립니다.")

    st.session_state.setdefault("histories", {})
    history = st.session_state["histories"].setdefault(profile, [])
    corpus, index = PROFILES[profile]
    if not corpus.exists() or not index.exists():
        st.error("선택한 문서의 검색 자료가 준비되지 않았습니다.")
        st.code(f"powershell -ExecutionPolicy Bypass -File .\\scripts\\build-rag.ps1 -Profile {profile}", language="powershell")
        st.stop()
    try:
        chunks = read_corpus(profile, corpus.stat().st_mtime_ns)
        pdf_data, total_pages, source_hash = pdf_info(PDF.stat().st_mtime_ns)
        if any(chunk["source_sha256"] != source_hash for chunk in chunks):
            raise ValueError("원본 PDF가 분석 당시 문서와 달라졌습니다.")
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"문서를 열 수 없습니다: {exc}")
        st.stop()
    with st.sidebar:
        st.caption(f"B2B Dashboard Manual · {total_pages}페이지")
        st.download_button("원본 PDF 내려받기", data=pdf_data, file_name=PDF.name,
                           mime="application/pdf", on_click="ignore", width="stretch")
        if st.button("대화 지우기", width="stretch"):
            history.clear()
            st.session_state.pop(f"record_{profile}", None)
            st.session_state[f"page_{profile}"] = 1
            st.rerun()

    st.title("질문하고, 원문으로 확인하세요")
    st.caption(f"{PROFILE_LABELS[profile]} · {device.upper()} · B2B Dashboard 매뉴얼")
    chat_column, evidence_column = st.columns([1.15, 1], gap="large")
    with chat_column:
        st.subheader("질문과 답변")
        messages = st.container(height=590, border=True)
        with messages:
            if not history:
                st.markdown("매뉴얼의 기능·메뉴·배점을 질문해보세요.")
                st.info("예: Spec Monitoring에서 이슈 필터를 초기화하려면 어떻게 하나요?")
            for record_index, record in enumerate(history):
                with st.chat_message("user"):
                    st.write(record["question"])
                with st.chat_message("assistant"):
                    st.markdown(record["answer"])
                    pages = list(dict.fromkeys(doc["page"] for doc in record["documents"] if not doc["skipped"]))
                    if pages:
                        buttons = st.columns(len(pages))
                        for column, page in zip(buttons, pages):
                            column.button(f"p.{page} 보기", key=f"source_{profile}_{record_index}_{page}",
                                          on_click=select_source, args=(profile, record_index, page))
                    st.caption(f"{PROFILE_LABELS[profile]} · {record['device'].upper()}")
        question = st.chat_input("매뉴얼에 대해 질문하세요", submit_mode="disable", max_chars=2000)
        if question and question.strip():
            record = {"question": question.strip(), "answer": "", "device": device, "documents": []}
            with messages:
                with st.chat_message("user"):
                    st.write(record["question"])
                with st.chat_message("assistant"):
                    try:
                        with resource_lock():
                            with st.spinner("매뉴얼에서 근거를 찾고 있습니다…"):
                                docs = retrieve(record["question"], top_k, index, device)
                                context, skipped = build_context(docs)
                                record["documents"] = [{"id": doc.id, "content": doc.page_content,
                                                          **doc.metadata, "skipped": doc.id in skipped} for doc in docs]
                            record["answer"] = st.write_stream(stream_chat(record["question"], context, device))
                    except Exception as exc:
                        record["answer"] = f"답변을 생성하지 못했습니다: {exc}"
                        st.error(record["answer"])
            history.append(record)
            select_source(profile, len(history) - 1, first_page(record))
            st.rerun()

    with evidence_column:
        st.subheader("근거 문서")
        current = None
        if history:
            record_index = st.selectbox("확인할 답변", range(len(history)), index=len(history) - 1,
                                       key=f"record_{profile}", on_change=select_record, args=(profile,),
                                       format_func=lambda i: f"{i + 1}. {history[i]['question'][:65]}")
            current = history[record_index]
        st.session_state.setdefault(f"page_{profile}", 1)
        page = st.number_input("원본 페이지", min_value=1, max_value=total_pages,
                               step=1, key=f"page_{profile}")
        source_pages = {doc["page"] for doc in current["documents"] if not doc["skipped"]} if current else set()
        if current and page not in source_pages:
            st.caption("현재 페이지는 직접 열어본 원문이며, 이 답변에 사용된 근거 페이지는 아닙니다.")
        original_tab, images_tab, text_tab = st.tabs(["원본 페이지", "추출 이미지", "인식된 내용"])
        with original_tab:
            try:
                st.image(page_preview(page, PDF.stat().st_mtime_ns), caption=f"원본 매뉴얼 · p.{page}", width="stretch")
            except Exception as exc:
                st.error(f"페이지를 표시하지 못했습니다: {exc}")
        with images_tab:
            images = {image["path"]: image for chunk in chunks if chunk["page"] == page for image in chunk.get("images", [])}
            if not images:
                st.info("이 페이지에서 별도로 추출한 이미지는 없습니다. 원본 페이지 탭에서 전체 화면을 볼 수 있습니다.")
            else:
                for number, image in enumerate(images.values(), 1):
                    try:
                        path = (ROOT / image["path"]).resolve()
                        path.relative_to(ROOT.resolve())
                        data = image_bytes(image["path"], image["sha256"], path.stat().st_mtime_ns)
                        st.image(data, caption=f"p.{page} · 추출 이미지 {number}", width="stretch")
                    except (OSError, ValueError, KeyError) as exc:
                        st.warning(f"추출 이미지를 표시하지 못했습니다: {exc}")
        with text_tab:
            page_chunks = [chunk for chunk in chunks if chunk["page"] == page]
            for chunk in page_chunks:
                title = chunk["section"] if chunk["section"] != "Page" else "페이지 전체"
                with st.expander(title, expanded=len(page_chunks) == 1):
                    st.markdown(display_text(chunk["content"]))
        if current and current["documents"]:
            with st.expander("검색된 근거 목록"):
                for doc in current["documents"]:
                    label = "답변 입력에서 제외" if doc["skipped"] else "답변 근거"
                    st.write(f"p.{doc['page']} · {doc['section']} · {label}")
        st.caption("이미지가 보이는 것과 답변 모델이 이미지를 직접 이해하는 것은 별개입니다.")


if __name__ == "__main__":
    main()
