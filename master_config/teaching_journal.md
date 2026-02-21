# Teaching Journal

*This journal is maintained by the master to track the student's progress, recurring issues, and interventions made.*

---

(No entries yet. The journey begins.)


---
**2026-02-21 17:29 UTC**

## Entry 1 - Artifact Leakage & Missing File System Tools
**Problem:** Student output contained raw formatting tokens ('BeNull', '<|im_start|>', '<think>' blocks) visible to the user. Additionally, the student had no file system tools and could only provide instructions when the user expected the folder to actually be created.
**Actions taken:** (1) Updated system prompt to explicitly forbid artifact leakage. (2) Added 'create_folder' tool so the student can actually create directories. (3) Retrying with hint to use the tool.
**Watch for:** Continued artifact leakage in future responses. Also watch if the student properly uses the new tool.