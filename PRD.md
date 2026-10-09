# Set Curator: product requirements

Status: living document, started 2026-10-06. It describes what exists, what it is for, and where it is going.
Personal data (song lists, filings, notes) is never part of this repository; see "Data and privacy".

## 1. Summary

Set Curator turns a DJ's own Rekordbox library into sets they can play. It listens to the audio, groups songs
by how they sound, and writes the result back into Rekordbox as folders, hot cues and comments, so the
organisation shows up on the decks (CDJ / XDJ) after a USB export. A review page lets the DJ confirm, correct
and annotate what the system proposes, and the system learns the DJ's own taste from those answers.

## 2. Users and problem

**Primary user:** a working DJ with a library of several hundred to a few thousand tracks, played on
Pioneer decks from a USB stick, who organises by feel rather than by genre tags.

**Problems today**
- Crates are grouped by where a song was found, not by how it sounds or where it fits in a set.
- Finding the next song means remembering the library; sound and energy are not searchable.
- Hot cues are set by hand, one track at a time.
- A DJ's taste is personal (for example "songs a saxophone could play over") and no generic tool models it.

## 3. Goals

1. **Organise by sound.** Songs land in folders by sound family, tempo and role, not by crate.
2. **Make sets easier to play.** Role folders (openers to closers), energy tiers, set arcs, bridges between
   crates, and tempo order that blends neighbouring songs.
3. **Show it on the deck.** Everything is written into Rekordbox (folders, hot cues A/B/C, comments), so it
   survives USB export without a laptop at the gig.
4. **Learn the DJ's taste.** Filings and notes the DJ gives become hints that shape every later curation.
5. **Stay safe.** Never lose or corrupt the DJ's library, whatever the tool does.

Non-goals: replacing Rekordbox, streaming or downloading music, mixing automatically, or sharing a library.

## 4. What exists

| Area | What it does |
|---|---|
| Analysis | Essentia `discogs-effnet` embeddings per song; section-by-section "palette"; genre, mood, voice and tonal models; high-end openness; bass and peak measurements; saxophone and melody measurements. All local, incremental. |
| Rekordbox write | Playlists and nested folders under one `SC` folder, hot cues A/B/C on the beatgrid, comment tags. |
| Organisation | Per-playlist sound and tempo folders; 8 set roles by tempo lane; set arcs by mood palette; cross-pollination (bridges, moods, relatives); a Dub & Reggae tree; Sax and Aura folders by energy tier. |
| Review GUI | Flask page: look up any song, hear a 30 s clip (or the whole track), file songs into groups, answer yes/no, add notes. Optional password-protected tunnel for use away from the computer. |
| Curator app | Seed-based curation across the library and cosine.club, with ledger and optional Spotify playlist. |

## 5. Taste maker (the next big capability)

**Idea.** The sax and aura review pages showed that the DJ's own filing is a better teacher than any score.
The taste maker generalises that loop for any group the DJ wants.

**The loop**
1. The DJ names a group and confirms a few songs that belong.
2. The system searches the library **by sound** (palette similarity to the confirmed songs). The same artist is a
   small bonus and never a constraint. Songs found by several independent searches are shown first.
3. The DJ reviews candidates in a page: file under groups, yes/no, notes.
4. The answers are stored; the next build uses them as ground truth, and the next search starts from more seeds.

**Rules the DJ has set**
- Their filing and notes outrank the system's scores.
- Notes are hints, read for role words ("good sustainer"), energy words ("low energy") and a few loose words
  ("driving", "laid back"). They nudge; the audio can still disagree.
- A taste can include songs that are not for mixing. A note such as "hard to mix" flags a song for listening:
  it can only take the Opener, Moment or Closer roles and is also listed in its own playlist.
- Categories with a different vibe (dub reggae) get their own tree, even inside other groups.
- A taste is not an energy level or a role: the same group can span many roles and tempos, so groups are
  sorted by energy tier and tempo inside.

**Findings so far (for tuning, not facts about any library)**
- Sound similarity finds the DJ's kind of song well (about 80% kept) but cannot tell which sub-group it belongs
  in; the DJ's filing decides that.
- Songs backed by two or more searches are kept more often than single-search ones.
- A saxophone detector alone misses many songs the DJ hears as sax; similarity to confirmed songs finds more.
- Measurements that sounded sensible can be wrong in direction ("room" as empty mid-range); check them against
  the DJ's answers before trusting them.

**Requirements**
- Review page works for any group, not just sax and aura, with notes and look-up.
- Candidates carry the reason they were offered.
- Answers and notes persist in local files and feed Set roles, folder builds and comments.
- A summary of what the system has learned is shown to the DJ in plain words, so they can correct it.

## 6. Functional requirements (existing behaviour to keep)

- **Folders:** everything generated lives under one `SC` folder in Rekordbox; rebuilding replaces a folder,
  never touches anything outside `SC`.
- **Comments:** `[OPEN|BAL|CLOSED | playlists | AURA | SAX tags] original comment`. Re-running replaces only
  the tag; the DJ's own text is kept.
- **Hot cues:** A at the first kick entry, B eight bars before the first drop, C at the peak. Re-running only
  replaces cues the tool wrote itself and the DJ has not moved.
- **Roles:** every song gets exactly one of eight roles from audio features within its tempo lane, plus the
  DJ's playlist leans, corrections and notes; reggae-world songs are barred from Momentum builders and Peak time.
- **Copies:** the same song stored as several files counts once.
- **Exclusions:** `excluded.json` removes songs (samples, wrong edits) from every analysis and folder.

## 7. Safety and constraints

- Rekordbox must be closed for any write; the tool refuses otherwise. Every write makes a timestamped backup of
  `master.db` first. Nothing deletes tracks or audio files; removal from the collection and from disk is the
  DJ's own action.
- The DJ's original files are never modified. Folders of originals and samples are off limits.
- USB export: Rekordbox writes track info (comments, cues) to a stick only for songs new to that stick, and never
  removes playlists the library no longer has. A clean update means deleting the old `SC` folders and tracks on
  the stick, then exporting again. Plan content changes so the stick needs this once.
- The stick may hold unrelated data; never format it.
- The review GUI is reachable from outside only through a tunnel and only with a password; on the computer
  itself it needs none.

## 8. Data and privacy

Personal files (`tracks.json`, embeddings, cues, filings, notes, answers, caches, `.env`) are git-ignored.
API keys and the GUI password live in `.env`. Nothing from the library leaves the computer except audio clips
served to the signed-in DJ through the tunnel, and Spotify or cosine.club queries the DJ asks for.

## 9. Success measures

- Share of proposed candidates the DJ keeps (today about 80%).
- Share of a new group the DJ can build in one review session without hunting through the library.
- No Rekordbox or stick problems traced to the tool (zero lost tracks, zero broken exports).
- Songs found at the deck: the DJ can open a role or taste folder and play without browsing.

## 10. Roadmap

1. Taste maker: generic groups, learned-taste summary, further search rounds from confirmed songs.
2. Spotify: a playlist of unknown songs from each set; an Aura (or any group) playlist on request.
3. Hot cue accuracy on the decks (known hit-and-miss cases) and a check against the deck's own analysis.
4. Name the sound groups in the DJ's words.
5. File Fetcher: stop the same song downloading once per crate.
6. Packaging: a single command that runs the incremental pipeline for new songs and rebuilds folders.

## 11. Open questions

- How should learned taste be shown so the DJ can correct a wrong lesson quickly?
- Should notes also feed search (for example "driving" as a sound query), or stay as role and energy hints?
- A permanent tunnel address versus a temporary one, and whether away-from-desk filing needs offline support.
- A clean way to push comment and cue updates to songs already on a stick without re-exporting them.
