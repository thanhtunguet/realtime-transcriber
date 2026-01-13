# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- **Improved deduplication in real-time transcription** - Replaced the simple character-based prefix matching algorithm with a sophisticated word-based overlap detection system.

### Fixed

- **Duplicate text output in refined transcriptions** - The sliding window refinement was outputting repeated content because the algorithm only compared against the last refined text instead of all committed text.

### Details

#### Problem

The previous `_diff_refined_text` method used character-level prefix matching and only tracked the last refined text. When the audio sliding window moved, the new refined transcription would contain overlapping content from previous outputs, causing duplicates like:

```
[21:46:28] Refined: relationships here, but you can apply this equally to
[21:46:31] Refined: to intercultural friends or even work-placing counters.
[21:46:34] Refined: here, but you can apply this equally to intercultural friends...
```

#### Solution

The new algorithm introduces:

1. **Cumulative committed text tracking** (`committed_text`) - Tracks ALL text that has been output to the user, not just the last refinement.

2. **Word-based overlap detection** (`_find_overlap_length`) - Finds how many words at the start of new refined text match the end of committed text:
   - Searches for overlaps from longest to shortest (up to 30 words)
   - Uses fuzzy matching with 70% threshold to handle minor transcription variations
   - Normalizes words (lowercase, strip punctuation) for better matching

3. **Duplicate content filter** - When no direct overlap is found, checks if 70%+ of words in the refined text already appeared in recent committed text (last 50 words), and skips output if so.

#### New Methods

- `_normalize_word(word)` - Normalizes a word for comparison by lowercasing and stripping punctuation
- `_find_overlap_length(committed_words, new_words)` - Finds word-level overlap between committed and new text

#### Expected Result

With the same audio input, the output should now be cleaner:

```
[21:46:28] relationships here, but you can apply this equally to
[21:46:31] Refined: intercultural friends or even work-placing counters.
[21:46:34] Refined: Oddly, these domains sometimes overlap.
[21:46:37] Refined: The first challenge I'd like to share with you today...
```

Each line now contains only the **new** portion of the transcription, avoiding repetition of previously output content.
