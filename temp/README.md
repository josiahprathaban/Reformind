# Reformind – Your Reformed AI Pastor

> **Biblical wisdom, Reformed truth.**  
> Reformind is an AI assistant trained on Scripture and historic Reformed confessions, designed to answer theological questions within the boundaries of Reformed doctrine.

## 📖 About

Reformind uses **Retrieval-Augmented Generation (RAG)** to provide answers grounded in:
- The Bible (public domain translations like KJV)
- The Five Solas
- The 95 Theses
- The Westminster Shorter Catechism
- The Heidelberg Catechism
- Other trusted Reformed writings

It's designed for:
- Pastoral guidance
- Theological study
- Catechism training
- Christian education

---

## ✨ Features

- Answers **strictly** from Scripture and Reformed theology  
- Cites sources for every response  
- Extensible – add your own theological documents  
- Web-based chat interface for ease of use  

---

## 🛠 Tech Stack

- **Backend:** FastAPI, LlamaIndex, HuggingFaceEmbedding  
- **Frontend:** Next.js, TailwindCSS  
- **Language Model:** Compatible with free, open-source models  
- **Deployment:** Vercel (frontend) + Railway/Render (backend)  

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/josiahprathaban/Reformind.git
cd Reformind
```

### 2. Backend Setup

```bash
cd backend
pip install -r requirements.txt
python indexer.py  # This creates the vector index (may take a few minutes)
uvicorn main:app --reload  # Start the backend server
```

### 3. Frontend Setup

In a new terminal window:

```bash
cd frontend
npm install
npm run dev  # Start the frontend development server
```

### 4. Access the Application

Open your browser and navigate to: [http://localhost:3000](http://localhost:3000)

### 5. Adding More Reformed Texts

To add additional Reformed texts:
1. Place text files in the `backend/data` folder
2. Update the `indexer.py` file to include the new sources
3. Re-run the indexer to update the vector database

---

## 🧠 How It Works

1. **Text Ingestion**: Reformed texts are processed and divided into semantic chunks
2. **Vector Embedding**: Each chunk is converted to a vector embedding using HuggingFace's sentence transformer
3. **Query Processing**: User questions are converted to the same vector space
4. **Retrieval**: The most relevant text chunks are retrieved based on semantic similarity
5. **Response Generation**: A structured response is generated using the retrieved information

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📧 Contact

Josiah Prathaban - [@josiahprathaban](https://github.com/josiahprathaban)

Project Link: [https://github.com/josiahprathaban/Reformind](https://github.com/josiahprathaban/Reformind)
