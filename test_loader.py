from core.pipeline import run_pipeline

result = run_pipeline(
    pdf_path="sample_docs/MYSQL_test.pdf",
    query="How do I install MySQL on Windows?"
)

print("\n" + "="*50)
print("FINAL ANSWER:")
print("="*50)
print(result["answer"])