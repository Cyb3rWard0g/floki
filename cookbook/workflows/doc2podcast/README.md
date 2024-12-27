# Doc2Podcast: Automating Podcast Creation from Research Papers

This workflow is a basic step toward automating the creation of podcast content from research using AI. It demonstrates how to process a single research paper, generate a dialogue-style transcript with LLMs, and convert it into a podcast audio file. While simple, this workflow serves as a foundation for exploring more advanced processes, such as handling multiple documents or optimizing content splitting for better audio output.

## Key Features and Workflow

* PDF Processing: Downloads a research paper from a specified URL and extracts its content page by page.
* LLM-Powered Transcripts: Transforms extracted text into a dialogue-style transcript using a large language model, alternating between a host and participants.
* AI-Generated Audio: Converts the transcript into a podcast-like audio file with natural-sounding voices for the host and participants.
* Custom Workflow: Saves the final podcast audio and transcript files locally, offering flexibility for future enhancements like handling multiple files or integrating additional AI tools.

## Prerequisites

* Python 3.8 or higher
* Required Python dependencies (install using `pip install -r requirements.txt`)
* A valid `OpenAI` API key for generating audio content
  * Set the `OPENAI_API_KEY` variable with your key value in an `.env` file.

## Configuration
To run the workflow, provide a configuration file in JSON format. The `config.json` file in this folder points to the following file "[Exploring Applicability of LLM-Powered Autonomous Agents to Solve Real-life Problems](https://github.com/OTRF/MEAN/blob/main/Rodriquez%20%26%20Syynimaa%20(2024).%20Exploring%20Applicability%20of%20LLM-Powered%20Autonomous%20Agents%20to%20Solve%20Real-life%20Problems.pdf)". Config example:

```json
{
    "pdf_url": "https://example.com/research-paper.pdf",
    "podcast_name": "AI Explorations",
    "host": {
        "name": "John Doe",
        "voice": "alloy"
    },
    "participants": [
        { "name": "Alice Smith" },
        { "name": "Bob Johnson" }
    ],
    "max_rounds": 4,
    "output_transcript_path": "podcast_dialogue.json",
    "output_audio_path": "final_podcast.mp3",
    "audio_model": "tts-1"
}
```

## Running the Workflow

* Place the configuration file (e.g., config.json) in the project directory.
* Run the workflow with the following command:

```bash
dapr run --app-id doc2podcast --resources-path components -- python3 workflow.py --config config.json
```

* Output:
  * Transcript: A structured transcript saved as `podcast_dialogue.json` by default. An example can be found in the current directory.
  * Audio: The final podcast audio saved as `final_podcast.mp3` as default. An example can be found [here](https://on.soundcloud.com/pzjYRcJZDU3y27hz5).

## Next Steps

This workflow is a simple starting point. Future enhancements could include:

* Processing Multiple Files: Extend the workflow to handle batches of PDFs.
* Advanced Text Splitting: Dynamically split text based on content rather than pages.
* Web Search Integration: Pull additional context or related research from the web.
* Multi-Modal Content: Process documents alongside images, slides, or charts.