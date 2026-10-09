"""Builds the pair list (import_audit.csv) and the three-panel page (review.html) for the import review.

The pairs: a song File Fetcher just imported (NEW) next to the file you already had (ON DISK), and singles that
only need a look against YouTube / Spotify. The reviewer's own page is copied and changed so the file already on
disk has its own panel next to the new file and the reference, instead of hiding behind a tab.
"""
import csv
import json
import os
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SC = HERE.parent
# File Fetcher (sff) is expected next to this repo; point SFF_APP_DIR at its "app" folder if it lives elsewhere
SFF_APP = Path(os.environ.get("SFF_APP_DIR", SC.parent / "File Fetcher" / "app"))
FF_PAGE = SFF_APP / "frontend" / "review.html"
PAIRS = [("278926005", "419691891"), ("806197853", "346064740"), ("412409593", "374167360"), ("351035136", "337958946"), ("233062893", "957295015")]
SINGLES = []   # Pandemonium (862261755) was a false alarm: no other copy, nothing to check


def build_csv():
    tr = {x["id"]: x for x in json.loads((SC / "tracks.json").read_text())}
    emb = np.load(SC / "embeddings.npz")
    vec = {str(i): v / np.linalg.norm(v) for i, v in zip(emb["ids"], emb["vectors"])}
    rows = []
    for new, old in PAIRS:
        if new in tr and old in tr:
            sim = float(vec[new] @ vec[old]) if new in vec and old in vec else ""
            rows.append((new, old, sim, "possible duplicate: same song as a file you already have"))
    for new in SINGLES:
        if new in tr:
            rows.append((new, "", "", "check: is it the song you want?"))
    with (HERE / "import_audit.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["playlist", "title", "verdict", "similarity_to_reference", "bass_share", "new_file", "reference"])
        w.writeheader()
        for new, old, sim, verdict in rows:
            t = tr[new]
            w.writerow({"playlist": (t["playlists"] or [""])[0], "title": t["title"], "verdict": verdict,
                        "similarity_to_reference": f"{sim:.3f}" if sim != "" else "", "bass_share": "",
                        "new_file": t["path"], "reference": tr[old]["path"] if old else ""})
    return rows, tr


def build_page():
    h = FF_PAGE.read_text(encoding="utf-8")

    def sub(old, new):
        nonlocal h
        assert old in h, "page changed upstream, cannot find: " + old[:60]
        h = h.replace(old, new, 1)

    sub("</style>", """    .rbinfo { margin-top: 10px; padding-top: 10px; border-top: 1px solid #22222e; font-size: 12px; color: #aaa; }
    .rbinfo .rbh { font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: #888; margin-bottom: 6px; }
    .rbinfo label { display: block; margin: 4px 0; color: #888; }
    .rbinfo input { width: 100%; margin-top: 2px; padding: 6px 8px; background: #14141c; color: #eee; border: 1px solid #2a2a38; border-radius: 6px; font: inherit; font-size: 13px; }
    .rbinfo input.edited { border-color: #ffd166; }
    .rbinfo .rbl { margin: 3px 0; word-break: break-word; }
    .rbinfo .rbl b { color: #ccc; font-weight: 500; }
    .rbinfo .rbs { min-height: 1.4em; color: #ffd166; }
    </style>""")
    sub(".panels { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }", ".panels { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }\n    #applyBtn, .finalrow, .restore { display: none !important; }")
    sub("@media (max-width: 800px) { .panels { grid-template-columns: 1fr; } }", "@media (max-width: 1000px) { .panels { grid-template-columns: 1fr; } }")
    sub('<div class="status" id="status-new"></div>', '<div class="status" id="status-new"></div>\n                <div class="rbinfo" id="rb-new"></div>')
    sub("<h2>New file · is this the right track?</h2>", "<h2>A · New file, just imported</h2>")
    sub('''            <div class="panel" id="panel-ref">
                <h2>Reference · the real track (always right)</h2>''', '''            <div class="panel" id="panel-old">
                <h2>B · File already on disk</h2>
                <div class="name">${it.has_old ? esc(it.old_name) : '<span class="missing">Nothing to compare: this song has no other copy</span>'}</div>
                ${it.has_old ? `<audio id="audio-old" controls preload="none" src="/api/review/audio/${it.id}/old"></audio>` : ''}
                <div class="status" id="status-old"></div>
                <div class="rbinfo" id="rb-old"></div>
            </div>
            <div class="panel" id="panel-ref">
                <h2>C · Reference, what it should sound like</h2>''')
    sub('''                    <button class="tab ${refTab === 'old' ? 'on' : ''}" data-tab="old">Old MP3</button>\n''', "")
    sub("try { refTab = localStorage.getItem('reviewRefTab') || 'yt'; } catch (e) {}", "try { refTab = localStorage.getItem('reviewRefTab') || 'yt'; } catch (e) {}\nif (refTab === 'old') refTab = 'yt';")
    sub("    if (n) hookAudio(n, 'new');", "    if (n) hookAudio(n, 'new');\n    const o = $('audio-old');\n    if (o) hookAudio(o, 'old');")
    sub("""    $('panel-new')?.classList.toggle('active', which === 'new');
    $('panel-ref')?.classList.toggle('active', which !== 'new');""", """    $('panel-new')?.classList.toggle('active', which === 'new');
    $('panel-old')?.classList.toggle('active', which === 'old');
    $('panel-ref')?.classList.toggle('active', which !== 'new' && which !== 'old');""")
    sub("function other() { return active === 'new' ? refKey() : 'new'; }", "function other() { const order = ['new', ...(view[pos]?.has_old ? ['old'] : []), refKey()]; return order[(order.indexOf(active) + 1) % order.length]; }")
    sub("function wire(it) {", """async function loadInfo(it) {
    let d;
    try { d = await (await fetch('/api/import/info/' + it.id)).json(); } catch (e) { return; }
    for (const side of ['new', 'old']) {
        const box = $('rb-' + side), r = d[side];
        if (!box || !r) continue;
        const f = (k, v) => r.edit && r.edit[k] != null ? r.edit[k] : v;
        box.innerHTML = `<div class="rbh">How Rekordbox has it</div>
            <label>Title <input data-f="title" value="${esc(f('title', r.title))}"></label>
            <label>Artist <input data-f="artist" value="${esc(f('artist', r.artist))}"></label>
            <div class="rbl"><b>Where</b> ${esc(r.where)}</div>
            <div class="rbl"><b>Format</b> ${esc(r.format)} · ${esc(r.length)} · ${esc(r.size)}</div>
            <div class="rbl"><b>Your playlists</b> ${esc(r.playlists.join(', ') || 'none')}</div>
            <div class="rbl"><b>Comment</b> ${esc(r.comment || '—')}</div>
            <div class="rbl"><b>Analysed</b> ${r.analysed ? 'yes · ' + r.cues + ' hot cues · in ' + r.sc_folders + ' SC folders' : 'no, not yet'} · <b>Added</b> ${esc(r.added)}</div>
            <div class="rbs"></div>`;
        const status = box.querySelector('.rbs');
        box.querySelectorAll('input').forEach(inp => {
            if (r.edit && r.edit[inp.dataset.f] != null && r.edit[inp.dataset.f] !== r[inp.dataset.f]) inp.classList.add('edited');
            inp.onchange = async () => {
                const body = { id: r.id, title: box.querySelector('[data-f=title]').value.trim(), artist: box.querySelector('[data-f=artist]').value.trim() };
                const same = body.title === r.title && body.artist === r.artist;
                await fetch('/api/import/edit', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...body, clear: same }) });
                box.querySelectorAll('input').forEach(i => i.classList.toggle('edited', i.value.trim() !== r[i.dataset.f]));
                status.textContent = same ? 'back to what Rekordbox has' : 'edit saved: it goes into Rekordbox when you tell me it is closed';
            };
        });
    }
}

function wire(it) {
    loadInfo(it);""")
    sub('<button class="btn primary" id="switchBtn"', """<button class="btn primary" data-to="new" title="Hear A at the same position">A · new <kbd>A</kbd></button>
            <button class="btn primary" data-to="old" title="Hear B at the same position">B · on disk <kbd>B</kbd></button>
            <button class="btn primary" data-to="ref" title="Hear C at the same position">C · reference <kbd>C</kbd></button>
            <button class="btn" id="switchBtn" """.rstrip())
    sub("Switch A/B <kbd>Tab</kbd>", "Next player <kbd>Tab</kbd>")
    sub("async function switchAB() {", """async function switchTo(which) {
    const target = which === 'ref' ? refKey() : which;
    if (target === 'old' && !view[pos]?.has_old) { toast('There is no other file to compare for this song.'); return; }
    if (target === active) return;
    const from = ctl(active), to = ctl(target);
    if (!to) { toast(target === 'sp' ? "Spotify's player can't be synced: press play in it yourself." : 'That player is not ready yet.'); return; }
    const t = from ? from.time() : 0, wasPlaying = from && from.playing();
    if (from) from.pause();
    setActive(target);
    await to.seek(t);
    if (wasPlaying) to.play();
}

async function switchAB() {""")
    sub("    $('switchBtn').onclick = switchAB;", "    $('switchBtn').onclick = switchAB;\n    document.querySelectorAll('[data-to]').forEach(b => b.onclick = () => switchTo(b.dataset.to));")
    sub("else if (e.key === 'u' || e.key === 'U') undo();", "else if (e.key === 'u' || e.key === 'U') undo();\n    else if (e.key.length === 1 && 'abcABC'.includes(e.key)) switchTo({ a: 'new', b: 'old', c: 'ref' }[e.key.toLowerCase()]);")
    sub("['right', '✓ New file is right', '1'],", "['right', '✓ Keep the new file (A)', '1'],")
    sub("['wrong', '✗ New file is wrong', '2'],", "['wrong', '✗ Remove the new file (A)', '2'],")
    sub("<title>", "<title>Import review · ") if "<title>" in h else None
    (HERE / "review.html").write_text(h, encoding="utf-8")


if __name__ == "__main__":
    rows, _ = build_csv()
    build_page()
    print(len(rows), "items;", sum(1 for r in rows if r[1]), "pairs")
