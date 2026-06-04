# Skunk TODO
- Human-in-the-loop for visual-understanding (chart-read) questions. We should write a mechanism that has the LLM identify visual understanding questions, and then surfaces a popup for the user to answer
- Data page cleaning -- define a pre-competition pipeline that cleans and prepares the data for a corpus before competition
- Clean up PageIndex build and search process, possibly use LLM-extracted dates on date mask for increased quality.
- Integrate PageIndex + sem_filter retrieval agent
- Investigate: revisions and how to deal with them (with sem filter this should actually be easy)
- Critique system redesign: previous critique system was removed because they caused failures over pedantic issues but did not catch meaningful bugs. We should probably come up with new critique subsystem.
- Integrate with Databricks grading harness

