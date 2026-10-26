"""Create a separate PM209 table JSONL with twelve image-grounded manual entries.

The existing Paddle output is read only. These entries are searchable transcriptions
of visible content, not a claim of exact cell-level accuracy.
"""

from __future__ import annotations

import hashlib
import html
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "PM209" / "data"
SOURCE = DATA / "paddle_tables_html_repaired"
OUTPUT = DATA / "paddle_tables_manual_20261026"
VALIDATION = SOURCE / "final-validation.json"


# Keys are the original table-region hashes from final-validation.json.
# Rows were checked against the twelve corresponding original crop images.
# Items with cropped edges, tiny symbols, or deliberately condensed prose are
# marked for review in the generated manifest.
SPECS = {
    "a09add8cd1a125908990fbc0": {
        "headers": ("Field", "Visible content"),
        "rows": [
            ("Name", "Define user-defined characters"),
            ("Format — ASCII", "ESC & y c1 c2 [x1 d1 ... d(y × x1)] ... [xk d1 ... d(y × xk)]"),
            ("Format — Hex", "1B 26 y c1 c2 [x1 d1 ...] ... [xk d1 ...]"),
            ("Format — Decimal", "27 38 y c1 c2 [x1 d1 ...] ... [xk d1 ...]"),
            ("Range", "y = 2; 32 ≤ c1 ≤ c2 ≤ 255; 0 ≤ x ≤ 14 (Font A); 0 ≤ x ≤ 12 (Font B); 0 ≤ d1 ... d(y × x) ≤ 255"),
            ("Description", "Defines user-defined characters. y specifies the number of bytes in the vertical direction. c1 is the beginning character code and c2 the final code; for one character c1 = c2. x is the number of dots in the horizontal direction."),
            ("Notes", "Consecutive character codes can be defined together. d is definition data for x horizontal dots from the left edge; unused dots on the right are spaces. A character definition requires y × x bytes. In definition data, 1 is printed and 0 is not printed. Downloaded definitions are possible for each font, selected by the ESC ! command. Definitions are cleared by ESC @, ESC ?, or printer reset/power-off."),
            ("Default", "The internal character set"),
            ("Reference", "ESC %, ESC ?"),
        ],
        "review": "Formula typography was normalized; verify x-index notation against the image before command-byte use.",
    },
    "774a578d569836ca65954fd0": {
        "headers": ("Field", "Visible content"),
        "rows": [
            ("Name", "Set horizontal tab positions"),
            ("Format — ASCII", "ESC D n1 ... nk NUL"),
            ("Format — Hex", "1B 44 n1 ... nk 00"),
            ("Format — Decimal", "27 68 n1 ... nk 0"),
            ("Range", "1 ≤ n ≤ 255; 0 ≤ k ≤ 32"),
            ("Description", "Sets horizontal tab positions. n is the column number counted from the beginning of the line. k is the number of positions to set."),
            ("Notes", "A position is character width × n from the line start. The width includes right-side spacing and doubles in double-width mode. This command replaces previously set tabs. n = 8 moves to the ninth digit when HT executes. Up to 32 positions are supported; supply n values in ascending order, then NUL. A value no greater than the preceding value finishes tab setting. ESC D NUL cancels all tabs. Later character-width changes do not move existing tab positions."),
            ("Default", "Tabs every eight characters: columns 9, 17, 25, ... for Font B (12 × 14)."),
            ("Reference", "HT"),
        ],
        "review": "Prose is condensed from the visible notes; command values need visual review before device control.",
    },
    "be988a1199e7ea6ff53a5599": {
        "headers": ("Field", "Visible content"),
        "rows": [
            ("Name", "Select justification"),
            ("Format — ASCII / Hex / Decimal", "ESC a n / 1B 61 n / 27 97 n"),
            ("Range", "0 ≤ n ≤ 2, 48 ≤ n ≤ 50"),
            ("Description", "Aligns all data in one line to the specified position."),
            ("n = 0 or 48", "Left justification"),
            ("n = 1 or 49", "Centering"),
            ("n = 2 or 50", "Right justification"),
            ("Notes", "The command is enabled only at the beginning of a line. Data skipped by HT is also subject to justification."),
            ("Default", "n = 0"),
        ],
        "review": "Image-grounded transcription of the visible table and adjacent command fields.",
    },
    "49cf36177e8f0f7e92317fbd": {
        "headers": ("Bit / field", "Visible content"),
        "rows": [
            ("Name", "Select paper detector(s) to output paper end signals"),
            ("Format — ASCII / Hex / Decimal", "ESC c 3 n / 1B 63 33 n / 27 99 51 n"),
            ("Range", "0 ≤ n ≤ 255"),
            ("Bit 0 off — 00 / 0", "Paper roll near end sensor disabled"),
            ("Bit 0 on — 01 / 1", "Paper roll near end sensor enabled"),
            ("Bit 1 off — 00 / 0", "Paper roll near end sensor disabled"),
            ("Bit 1 on — 02 / 2", "Paper roll near end sensor enabled"),
            ("Bit 2 off — 00 / 0", "Paper roll end detector disabled"),
            ("Bit 2 on — 04 / 4", "Paper roll end detector enabled"),
            ("Bit 3 off — 00 / 0", "Paper roll end detector disabled"),
            ("Bit 3 on — 08 / 8", "Paper roll end detector enabled"),
            ("Bits 4–7", "Undefined"),
            ("Notes", "Multiple detectors may be selected. A signal is output if any selected detector senses paper end. Switching occurs when the command executes and may be delayed by the receive-buffer state."),
            ("Default", "n = 15"),
        ],
        "review": "Source image says bit 3 'enable' in one cell; wording normalized to enabled.",
    },
    "5d4f266b5c22c326deeef0c2": {
        "headers": ("Main / outdoor code", "Check code name; judging device; visible auxiliary detail"),
        "rows": [
            ("E01", "Communication error between indoor and remote controller, detected at remote controller side; Remote controller"),
            ("E02", "Remote controller transmission error; Remote controller"),
            ("E03", "Communication error between indoor and remote controller, detected at indoor side; Indoor"),
            ("E04", "Communication circuit error between indoor/outdoor, detected at indoor side; Indoor"),
            ("E06 / E06", "Decrease of number of indoor units; I/F. Auxiliary: number of indoor units from which signal is normally received."),
            ("— / E07", "Communication circuit error between indoor/outdoor, detected at outdoor side; I/F"),
            ("E08 / E08", "Duplicated indoor addresses; Indoor / I/F"),
            ("E09", "Duplicated main remote controllers; Remote controller"),
            ("E10", "Communication error between indoor MCU; Indoor"),
            ("E12 / E12", "Automatic address start error; I/F. Auxiliary 01: indoor/outdoor communication; 02: communication between outdoor units."),
            ("E15 / E15", "Indoor is nothing during automatic addressing; I/F"),
            ("E16 / E16", "Capacity over / number of connected indoor units; I/F. Auxiliary 00: capacity over; 01: number of connected units."),
            ("E18", "Communication error between indoor units; Indoor"),
            ("E19 / E19", "Outdoor header units quantity error; I/F. Auxiliary 00: header is nothing; 02: two or more header units."),
            ("E20 / E20", "Other line connected during automatic address; I/F. Auxiliary 01: outdoor of other line connected; 02: indoor of other line connected."),
            ("E23 / E23", "Sending error in communication between outdoor units; I/F"),
            ("E25 / E25", "Duplicated follower outdoor addresses; I/F"),
            ("E26 / E26", "Decrease of number of connected outdoor units; I/F. Auxiliary: number of outdoor units whose signal is normally received."),
            ("E28 / E28", "Follower outdoor unit error; I/F. Auxiliary: detected outdoor unit number."),
            ("E31 / E31", "IPDU communication error; I/F. Auxiliary 01: IPDU1 error; 02: IPDU2 error; 03: IPDU1 and 2 error; 04: fan IPDU error; 05: IPDU + fan IPDU error; 06: IPDU2 + fan IPDU error; 07: all IPDU errors."),
        ],
        "review": "Code-to-name relationships transcribed; sensor-block lamp icons and some auxiliary display details are not represented. Review before using to diagnose equipment.",
    },
    "a6b2cacd190460e8df056984": {
        "headers": ("Main / outdoor code", "Check code name; judging device; visible auxiliary detail"),
        "rows": [
            ("P01", "Indoor fan motor error; Indoor"),
            ("P03 / P03", "Discharge temperature TD1 error; I/F"),
            ("P04 / P04", "High-pressure SW system operation; IPDU. Auxiliary 01: compressor 1 side; 02: compressor 2 side."),
            ("P05 / P05", "Phase-missing detection / phase error; I/F. Auxiliary 01: phase-missing detection; 02: phase error."),
            ("P07 / P07", "Heat sink overheat error; IPDU, I/F. Auxiliary 01: compressor 1 side; 02: compressor 2 side."),
            ("P10 / P10", "Indoor overflow error; Indoor. Auxiliary: detected indoor address."),
            ("P12", "Indoor fan motor error; Indoor"),
            ("P13 / P13", "Outdoor liquid back detection error; I/F"),
            ("P15 / P15", "Gas leak detection; I/F. Auxiliary 01: TS condition; 02: TD condition."),
            ("P17 / P17", "Discharge temperature TD2 error; I/F"),
            ("P19 / P19", "4-way valve inverse error; I/F. Auxiliary: detected outdoor unit number."),
            ("P20 / P20", "High-pressure protective operation; I/F"),
            ("P22 / P22", "Outdoor fan IPDU error; IPDU. Auxiliary includes IGBT short, fan motor position detective circuit error, fan motor trouble, TH sensor temperature error (heat sink overheat), TH sensor error and Vdc output error."),
            ("P26 / P26", "G-TR short protection error; IPDU. Auxiliary 01: compressor 1 side; 02: compressor 2 side."),
            ("P29 / P29", "Compressor position detective circuit system error; IPDU. Auxiliary 01: compressor 1 side; 02: compressor 2 side."),
            ("P31 / P31", "Other indoor unit error (group terminal unit error); Indoor"),
            ("—", "Error in indoor group; AI-NET. Wireless remote controller displays 'By alarm device ALT'."),
        ],
        "review": "Code-to-name relationships transcribed; lamp icons and the detailed P22 auxiliary bit mapping are not represented.",
    },
    "1d6c690960764280fc988c7a": {
        "headers": ("Error displayed", "Visible solution"),
        "rows": [
            ("door / Ed", "Close the door."),
            ("E1", "Make sure the source water faucet is turned on. Check the water pressure."),
            ("E2", "Clean the debris filter. Make sure the drainage hose is installed correctly."),
            ("E4", "Laundry load is unbalanced; loosen tangled laundry. Washing a single item such as a bathrobe or jeans may produce an unsatisfactory final spin and an E4 message."),
            ("E9", "Check whether the end of the drain hose is too low and whether it is submerged in water. Make sure the holder on the back is intact."),
        ],
        "review": "The image crop cuts off text after the E9 section; this transcription covers only visible lines. 'door' and 'Ed' appear together in the source row.",
    },
    "a69fd8b3c748d44e409fa95f": {
        "headers": ("Button / item", "Visible function"),
        "rows": [
            ("24. Up-Down Left-Right buttons", "Move horizontally or vertically between menu items and adjust selected menu values."),
            ("25. EXIT", "Exit the menu screen."),
            ("26. SRS", "Select SRS TruSurround XT mode."),
            ("27. MagicInfo", "MagicInfo Quick Launch Button."),
            ("28. P.MODE", "Display the current picture mode. AV/HDMI/TV modes: Dynamic → Standard → Movie → Custom. PC/DVI/MagicInfo MagicBright modes: Entertain → Internet → Text → Custom."),
            ("29. DUAL/MTS", "DUAL: select stereo/mono, Dual I / Dual II, and Mono/NICAM modes according to broadcast type. MTS: select Multichannel Television Stereo mode. FM Stereo audio defaults: Mono → Mono (manual change); Stereo → Mono ↔ Stereo; SAP → Mono ↔ SAP (default Mono)."),
            ("30. PIP", "A PIP screen appears when pressed; the source says this function does not work for this LCD display."),
            ("31. SWAP", "Swap PIP and main image; the source says this function does not work for this LCD display."),
        ],
        "review": "Button text is condensed, and small icon glyphs are omitted. The source embeds a small audio-mode table inside item 29.",
    },
    "ecab6565fc2f57fd4d73209e": {
        "headers": ("Specification", "Visible value"),
        "rows": [
            ("Clock", "Quartz locked"),
            ("Power back up", "Approx. 1 day or less"),
            ("Power requirements", "230 V AC, 50 Hz"),
            ("Power consumption — KV-V1430K/P", "60 W"),
            ("Power consumption — KV-V2110K/P", "81 W"),
            ("Operating temperature", "5°C to 40°C (41°F to 104°F)"),
            ("Storage temperature", "−20°C to 60°C (−4°F to 140°F)"),
            ("Dimensions — KV-V1430K/P", "Approx. 391 × 409 × 443 mm (w/h/d); 15½ × 16⅛ × 17½ inches"),
            ("Dimensions — KV-V2110K/P", "Approx. 540 × 518 × 470 mm (w/h/d); 21⅜ × 20½ × 18⅝ inches"),
            ("Mass — KV-V1430K/P", "Approx. 15 kg (33 lb 1 oz)"),
            ("Mass — KV-V2110K/P", "Approx. 25 kg (55 lb 2 oz)"),
        ],
        "review": "Inch fractions are small in the crop; recheck those figures against the original page before using exact dimensions.",
    },
    "57197db00a2580859d1735f2": {
        "headers": ("Programme", "Function"),
        "rows": [
            ("Well being", "Can aid physical and mental relaxation of vehicle occupants. A back massage with local warmth provides relaxation. Air is purified with ionisation and a pleasant fragrance. Calming music and pleasant lighting are used."),
            ("Power Nap", "Three phases: sleep onset (relaxing music and warm-seat massage); sleep (active functions deactivated or restricted as much as possible); waking (slightly stimulating music, ionisation and scent, seat ventilation and stimulating light)."),
            ("Training", "Counteracts muscle tension, limbs falling asleep or stress with targeted relaxation or stimulation exercises shown in short videos. Ionisation, a pleasant fragrance and suitable lighting are used."),
        ],
        "review": "Condensed wording retains the visible programme-to-function associations.",
    },
    "2b624e503e3855c3cdb9b027": {
        "headers": ("Warning / indicator lamp", "Possible cause, consequence and solution"),
        "rows": [
            ("Restraint system warning lamp", "The red lamp is lit while the drive system is running. The restraint system is malfunctioning (source refers to page 31). DANGER: fatal injury risk. Components may trigger unintentionally or fail in an accident; seat belt tensioners, airbags and the high-voltage on-board electrical system may be affected. Touching damaged high-voltage parts risks electrocution. Have the restraint system checked and repaired immediately at a qualified specialist workshop. After an accident, switch off the ignition immediately. Drive carefully, note multifunction-display messages, and consult a qualified specialist workshop immediately."),
        ],
        "review": "Warning paragraph condensed; safety-critical use requires checking the original image.",
    },
    "3126c1fa50a63546d49e3971": {
        "headers": ("Connector / item", "Visible specification"),
        "rows": [
            ("Pr/R/R-Y, Y/G/Y, Pb/B/B-Y", "BNC type (3). HD component video: Y (100% white) 0.7 Vp-p; Pr/Pb (75% color bar) 0.7 Vp-p, 75 Ω. HD RGB: R/G/B (100% white) 0.7 Vp-p, 75 Ω. SD RGB: R/G/B (100% white) 0.7 Vp-p, 75 Ω. SD component video: Y (100% white) 0.714 Vp-p; Pr/Pb (75% color bar) 0.756 Vp-p, 75 Ω."),
            ("VBS1, 2", "BNC type (2); VBS 1.0 Vp-p, 75 Ω"),
            ("PIX", "BNC type (1); VBS/R/G/B (VBS 1.0 Vp-p, 75 Ω)"),
            ("SYNC", "BNC type (1). HD: BTA-S001A tri-level sync, 0.6 Vp-p, 75 Ω. SD: composite sync, 0.3 Vp-p, 75 Ω. HD SYNC / SD SYNC selectable."),
            ("AUDIO OUTPUT CH-1, CH-2", "XLR 3-pin, male (2), 0/−20 dBu"),
            ("S-VIDEO OUTPUT", "4-pin"),
            ("HDMI OUTPUT", "Type A, 19-pin"),
            ("Supplied accessories", "Number plates (1 set); Operating Instructions: Japanese (1), English (1); Warranty booklet (1); CD-ROM (1)"),
            ("Optional accessories", "United States and Canada: Plug holder B (2-990-242-01), power cord set (1-551-812-XX). Other areas: Plug holder C (3-613-640-01), power cord set (1-782-929-XX). CCA-5-3 (3 m), CCA-5-10 (10 m) connection cables; CCZ-A5/10/25/50/100 multi-core cable; CCZZ-1B and CCZZ-1E cable extension connectors; service manual."),
            ("Related equipment", "HD Color Camera HXC-D70; RCP-1000-series Remote Control Panel"),
        ],
        "review": "The crop is a compound list of connectors, accessories and equipment; rows group visible related items. Exact part numbers should be rechecked visually.",
    },
}


def render_table(spec: dict) -> str:
    heads = "".join(f"<th>{html.escape(h)}</th>" for h in spec["headers"])
    rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in row) + "</tr>"
        for row in spec["rows"]
    )
    return f"<table><thead><tr>{heads}</tr></thead><tbody>{rows}</tbody></table>"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_review_index(manifest: list[dict]) -> None:
    sections = []
    for item in manifest:
        key = item["key"]
        src = f"../paddle_tables_html/assets/{item['split']}/{key}/image.png"
        sections.append(
            "<section><h2>" + html.escape(item["table_id"]) + "</h2>"
            "<p><strong>수기 요약 전사 · 재검수 필요:</strong> " + html.escape(item["review"]) + "</p>"
            "<div class='pair'><img src='" + src + "' alt='원본 표 영역'>"
            "<div class='result'>" + render_table(SPECS[key]) + "</div></div></section>"
        )
    page = (
        "<!doctype html><html lang='ko'><meta charset='utf-8'><title>PM209 남은 표 12개 검수</title>"
        "<style>body{font:15px/1.45 sans-serif;margin:24px;color:#222}section{border-top:2px solid #777;padding:18px 0}"
        ".pair{display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:start}img{max-width:100%;height:auto}"
        ".result{overflow:auto}table{border-collapse:collapse;width:100%}th,td{border:1px solid #aaa;padding:6px;vertical-align:top}"
        "th{background:#eee}@media(max-width:950px){.pair{grid-template-columns:1fr}}</style>"
        "<h1>PM209 남은 표 12개 수기 보강 검수</h1>"
        "<p>왼쪽은 원본 영역, 오른쪽은 검색용 HTML입니다. 긴 문장은 줄여 적었고 일부 아이콘은 빠져 있습니다."
        " 내용과 행·열의 완전한 정확도를 인증한 결과가 아닙니다.</p>" + "".join(sections) + "</html>"
    )
    (OUTPUT / "review.html").write_text(page, encoding="utf-8")


def main() -> None:
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    unresolved = {item["table_id"]: item for item in validation["unresolved"]}
    assert {v["key"] for v in unresolved.values()} == set(SPECS), "Unresolved IDs differ from manual entries"
    assert validation["added_tables"] == 2308
    if OUTPUT.exists():
        raise SystemExit(f"Output already exists; refusing to overwrite: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    seen = set()
    manifest = []
    try:
        for split in ("train", "val", "test"):
            src = SOURCE / f"{split}.jsonl"
            dst = OUTPUT / f"{split}.jsonl"
            with src.open("r", encoding="utf-8") as reader, dst.open("w", encoding="utf-8", newline="\n") as writer:
                for line in reader:
                    page = json.loads(line)
                    changed = False
                    for box in page["bounding_boxes"]:
                        item = unresolved.get(box["id"])
                        if item is None:
                            continue
                        assert item["split"] == split
                        assert box["structure"] == "Table" and not box.get("table_content")
                        key = item["key"]
                        assert key not in seen
                        image = DATA / "paddle_tables_html" / "assets" / split / key / "image.png"
                        assert image.is_file(), image
                        content = render_table(SPECS[key])
                        box["table_content"] = content
                        seen.add(key)
                        changed = True
                        manifest.append({
                            "split": split,
                            "page_id": page["id"],
                            "table_id": box["id"],
                            "key": key,
                            "source_image": str(image.relative_to(ROOT)),
                            "source_image_sha256": sha256(image),
                            "status": "manual_condensed_review_required",
                            "row_count": len(SPECS[key]["rows"]),
                            "review": SPECS[key]["review"],
                        })
                    writer.write(json.dumps(page, ensure_ascii=False, separators=(",", ":")) + "\n" if changed else line)
        assert seen == set(SPECS), f"Missing: {set(SPECS) - seen}"
        (OUTPUT / "manual-review.json").write_text(
            json.dumps({"method": "Original image visual transcription and condensation; review required before treating as exact cell transcription", "entries": manifest}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for item in manifest:
            (OUTPUT / f"{item['key']}.html").write_text(render_table(SPECS[item["key"]]) + "\n", encoding="utf-8")
        write_review_index(manifest)
        print(json.dumps({"output": str(OUTPUT), "manual_entries": len(manifest), "source_sha256": {s: sha256(SOURCE / f"{s}.jsonl") for s in ("train", "val", "test")}}, ensure_ascii=False))
    except BaseException:
        # Keep already-written files for diagnosis, but remove partial split JSONLs
        # so they cannot be mistaken for complete results.
        for split in ("train", "val", "test"):
            (OUTPUT / f"{split}.jsonl").unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    if sys.argv[1:] == ["--review-only"]:
        existing = json.loads((OUTPUT / "manual-review.json").read_text(encoding="utf-8"))
        write_review_index(existing["entries"])
        print(OUTPUT / "review.html")
    else:
        main()
