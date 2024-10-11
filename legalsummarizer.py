from transformers import pipeline
# Initialize the summarization pipeline with a pre-trained model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
# Sample legal document for demonstration
legal_text = """
    This Agreement, dated as of the date hereof, is made and entered into by and among the parties hereto, 
    in connection with and pursuant to the laws of the State. The parties agree to the following terms and conditions.
    In the event of a dispute, the parties shall seek arbitration before taking legal action in any court of law.
    Moreover, this agreement is subject to termination upon the occurrence of the specified conditions stated herein.
"""

# Generate the summary
summary = summarizer(legal_text, max_length=130, min_length=30, do_sample=False)

# Print the summary
print("Summary of the legal document:\n", summary[0]['summary_text'])
# Helper function to split document into smaller chunks
def split_text(text, chunk_size=1024):
    """
    Splits the input text into chunks of a specified size.
    Args:
        text: The input text to split.
        chunk_size: The maximum size of each chunk.
    Returns:
        List of text chunks.
    """
    chunks = []
    while len(text) > chunk_size:
        split_point = text[:chunk_size].rfind(".") + 1
        if split_point == 0:
            split_point = chunk_size
        chunks.append(text[:split_point])
        text = text[split_point:]
    chunks.append(text)
    return chunks

# Split the legal document and summarize each chunk
legal_chunks = split_text(legal_text)
summaries = [summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text'] for chunk in legal_chunks]

# Combine and display the summary
full_summary = " ".join(summaries)
print("Combined Summary of the legal document:\n", full_summary)
# Initialize with a different model, such as T5
summarizer_t5 = pipeline("summarization", model="t5-small")

# Generate a summary using T5
summary_t5 = summarizer_t5(legal_text, max_length=130, min_length=30, do_sample=False)
print("Summary using T5:\n", summary_t5[0]['summary_text'])
