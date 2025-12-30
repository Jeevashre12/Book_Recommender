import pandas as pd
import numpy as np
from dotenv import load_dotenv
import gradio as gr
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DEFAULT_COVER_URL = "img.png"

books = pd.read_csv("books_with_emotions.csv")

books["thumbnail"] = books["thumbnail"].astype(str)
books["thumbnail"] = np.where(
    books["thumbnail"].str.endswith(","),
    books["thumbnail"].str[:-1],
    books["thumbnail"]
)

CATEGORY_COL = "simple_categories" if "simple_categories" in books.columns else "categories"

documents = []
for _, row in books.iterrows():
    documents.append(
        Document(
            page_content=str(row["description"]),
            metadata={"isbn13": row["isbn13"]}
        )
    )

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma.from_documents(docs, embeddings)

def get_large_thumbnail(url):
    if not isinstance(url, str):
        return DEFAULT_COVER_URL

    url = url.strip()

    if url == "" or url.lower() == "nan":
        return DEFAULT_COVER_URL

    if not url.startswith("http"):
        return DEFAULT_COVER_URL

    if "zoom=" in url:
        return url.replace("zoom=1", "zoom=3")

    return url

def retrieve_semantic_recommendations(
    query,
    category="All",
    tone="All",
    initial_top_k=50,
    final_top_k=16,
):
    recs = db.similarity_search_with_score(query, k=initial_top_k)
    isbn_list = [doc.metadata["isbn13"] for doc, _ in recs]

    book_recs = books[books["isbn13"].isin(isbn_list)]

    if category != "All":
        book_recs = book_recs[book_recs[CATEGORY_COL] == category]

    if tone == "Happy":
        book_recs = book_recs.sort_values(by="joy", ascending=False)
    elif tone == "Surprising":
        book_recs = book_recs.sort_values(by="surprise", ascending=False)
    elif tone == "Angry":
        book_recs = book_recs.sort_values(by="anger", ascending=False)
    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values(by="fear", ascending=False)
    elif tone == "Sad":
        book_recs = book_recs.sort_values(by="sadness", ascending=False)

    return book_recs.head(final_top_k)

def recommend_books(query, category, tone):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        short_desc = " ".join(str(row["description"]).split()[:30]) + "..."

        authors = str(row["authors"]).split(";")
        if len(authors) == 2:
            author_str = f"{authors[0]} and {authors[1]}"
        elif len(authors) > 2:
            author_str = f"{', '.join(authors[:-1])}, and {authors[-1]}"
        else:
            author_str = authors[0]

        caption = f"{row['title']} by {author_str}: {short_desc}"

        image_url = get_large_thumbnail(row["thumbnail"])
        results.append((image_url, caption))

    return results

categories = ["All"] + sorted(books[CATEGORY_COL].dropna().unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(
            label="Describe the book you want",
            placeholder="e.g., A story about forgiveness and personal growth"
        )
        category_dropdown = gr.Dropdown(choices=categories, label="Category", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Emotional Tone", value="All")
        submit_button = gr.Button("Find Recommendations")

    output = gr.Gallery(columns=4, rows=4)

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )

if __name__ == "__main__":
    dashboard.launch()
