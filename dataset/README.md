# Shakespeare SLM/RAG Teaching Dataset

This package contains a small, instructor-curated dataset for CSCI433/933 Assignment 2: Domain Adaptation with Small Language Models.

## Contents

Each required play is provided in a separate JSON file:

- `hamlet.json`
- `macbeth.json`
- `romeo_and_juliet.json`

For convenience, each play is also provided in two retrieval-friendly JSONL formats:

- `*_scene_chunks.jsonl`: one record per scene, suitable as a baseline retrieval unit.
- `*_utterances.jsonl`: one record per utterance or stage direction, suitable for finer-grained retrieval or custom chunking.

The file `instructor_questions.json` contains instructor-provided evaluation questions.

## Structure

Each main JSON file has this structure:

```json
{
  "metadata": { "...": "..." },
  "scenes": [
    {
      "scene_id": "macbeth_1_3",
      "play": "Macbeth",
      "act": 1,
      "scene": 3,
      "location": "A heath",
      "scene_summary": "...",
      "keywords": ["prophecy", "ambition"],
      "utterances": [
        {
          "speaker": "MACBETH",
          "speaker_original": "MACBETH",
          "text": "So foul and fair a day I have not seen.",
          "source_id": "macbeth_1_3_0001"
        }
      ],
      "text": "...scene text..."
    }
  ]
}
```

## Recommended Student Use

Students may use the scene-level files directly for RAG retrieval, or they may build their own chunks from the utterance-level files. Scene-level chunks are easier and more robust; utterance-level records allow more experimentation.

## Source

The base texts were obtained from Project Gutenberg:

- Hamlet: https://www.gutenberg.org/cache/epub/1787/pg1787.txt
- Macbeth: https://www.gutenberg.org/cache/epub/1795/pg1795.txt
- Romeo and Juliet: https://www.gutenberg.org/cache/epub/1777/pg1777.txt

Project Gutenberg provides access and distribution terms on its website. Users should comply with those terms and with local copyright requirements. This dataset is intended for educational use in the assignment context.

## Dataset Statistics

| Play | Scenes | Utterance/stage records |
|---|---:|---:|
| Hamlet | 20 | 1657 |
| Macbeth | 28 | 830 |
| Romeo and Juliet | 25 | 1126 |

## Notes for Instructors

The scene summaries and keywords are short teaching aids designed to support students who have not studied Shakespeare before. They should not be treated as ground truth literary criticism. Students should still evaluate retrieval and generated answers against the source passages.

## Known Limitations

The parsing is intentionally lightweight. Stage directions and speaker labels may not perfectly match scholarly editions. This is acceptable for the assignment because the main learning objectives are retrieval design, grounding, model adaptation, and evaluation.
