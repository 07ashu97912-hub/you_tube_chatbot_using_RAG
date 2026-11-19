from youtube_transcript_api import YouTubeTranscriptApi , TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableParallel , RunnablePassthrough , RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.messages import SystemMessage,HumanMessage,AIMessage
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_core.documents import Document
from dotenv import load_dotenv
import streamlit as st
import re
import bisect
import time

load_dotenv()

def extract_id(link):
    link = link.strip()
    if re.fullmatch(r'[A-Za-z0-9_-]{11}',link):
        return link
    pattern = r'(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^"&?\/\s]{11})'
    match = re.search(pattern, link, re.IGNORECASE)
    return match.group(1) if match else None

model = ChatGroq(
    model="openai/gpt-oss-20b"
)

parser = StrOutputParser()

link = st.text_input('enter the you tube video ID : ')
video_id = extract_id(link)
user_input = st.text_input('enter your query : ')

multi_query_generation_prompt = ChatPromptTemplate([
    ('system','''
            You are an AI assistant specialized in query generation
        for a YouTube video-based RAG application.

        Your task is to generate 4 different versions of the
        user's question to improve document retrieval.

        Follow these instructions:

        1. Generate 4 alternative queries that capture the
           original question from different perspectives.

        2. Use different terminology, keywords and sentence
           structures where appropriate.

        3. Preserve the original intent and meaning of the query.

        4. Avoid generating repetitive or unrelated questions.

        5. Do not answer the question.

        6. Each generated query should be independently
           understandable.

        7. Return each query on a separate line.

        8. Do not include numbering, explanations or any
           additional text.'''),
    ('human','{message}')
])

if st.button('explain' ):
    try:
        youtubeapi = YouTubeTranscriptApi()
        transcript_list = youtubeapi.fetch(video_id = video_id,languages = ['en'])
        transcript = ' '.join(chunk.text for chunk in transcript_list)

        transcript_all_data = [
            Document(page_content=item.text,
                     metadata = {
                        'source':video_id,
                        'start':item.start,
                        'duration':item.duration
                     },)for item in transcript_list
        ]

    except TranscriptsDisabled:
        print('no caption available for this video')

    def convert_seconds(total_seconds):
        hours , remainder = divmod(total_seconds , 3600)
        minutes , seconds = divmod(remainder , 60)
        return f'{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}'

    def split_with_metadata(
            docs: list[Document] , chunk_size : int = 1000 , chunk_overlap : int = 20 
    ) -> list[Document]:
        full_parts = []
        char_offsets = []
        timestamps = []
        current_pos = 0

        for doc in docs:
            text = doc.page_content.strip()
            if not text:
                continue
            char_offsets.append(current_pos)
            start_time = doc.metadata.get('start',0.0)
            end_time = round(start_time + doc.metadata.get('duration',0.0),2)
            timestamps.append((start_time , end_time))

            full_parts.append(text)
            current_pos += len(text) + 1

        full_text = ' '.join(full_parts)
        source_id = doc.metadata.get('source','')

        splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            add_start_index = True
        )
        raw_chunks = splitter.create_documents([full_text],metadatas=[{'source':source_id}])

        final_chunks = []
        for chunks in raw_chunks:
            c_start = chunks.metadata['start_index']
            c_end = c_start + len(chunks.page_content)

            start_idx = max(0 , bisect.bisect_right(char_offsets , c_start) - 1)
            end_idx = max(0 , bisect.bisect_left(char_offsets , c_end) - 1)
            chunk_start_sec = timestamps[start_idx][0]
            chunk_end_sec = timestamps[end_idx][1]

            final_chunks.append(
                Document(
                    page_content=chunks.page_content,
                    metadata={
                    "source": source_id,
                    "start": convert_seconds(chunk_start_sec),
                    "end": convert_seconds(chunk_end_sec),
                    "duration": round(chunk_end_sec - chunk_start_sec, 2)
                    }
                )
            )

        return final_chunks
    
    chunks = split_with_metadata(transcript_all_data)

    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_store = FAISS.from_documents(chunks , embedding)

    retriever = vector_store.as_retriever(search_type = 'mmr' , search_kwargs = {'k':3 , 'lambda_mult':0.8})

    multi_query_generation_chain = multi_query_generation_prompt | model | parser
    multi_query = multi_query_generation_chain.invoke({'message':user_input})
    multi_query_list = [ q.strip() for q in multi_query.split('\n') if q.strip()]
    multi_query_list.append(user_input)

    all_docs = []
    for query in multi_query_list:
        docs = retriever.invoke(query)
        all_docs.extend(docs)

    redundant_filter = EmbeddingsRedundantFilter(
        embeddings = embedding,
        similarity_threshold = 0.90
    )

    context = redundant_filter.transform_documents(all_docs)

    answer_prompt = ChatPromptTemplate([
        ('system','''
                You are an educational assistant designed to answer questions based on YouTube video transcripts.
                Your task is to provide accurate, concise, and well-structured answers using ONLY the information available in the provided transcript context.

                Instructions:

                1.Answer the user's question directly and maintain the original context of the discussion.
                2.Combine information from multiple transcript chunks when necessary to construct a complete answer.
                3.Do not introduce external knowledge, assumptions, or information that is not supported by the transcript.
                4.If the provided context does not contain sufficient information to answer the question, explicitly state that the video transcript does not provide enough information.
                5.Avoid repeating information or including unnecessary explanations.
                6.Use headings, numbered lists, or tables whenever they improve readability.

                Timestamp attribution:

                1.Every factual point or step in your answer must have its corresponding source timestamp.
                2.Extract timestamps exclusively from the metadata associated with the retrieved transcript chunks.
                3.Represent timestamps in minutes and seconds (MM:SS).
                4.If information comes from multiple chunks, include all relevant timestamp ranges.
                5.Never invent, estimate, or assume timestamps.
                6.Do not include a separate introduction, conclusion, or unrelated information unless necessary to answer the question.
        transcript context : {context}
            '''),
        ('human','{question}')
        ])

    context_with_metadata = []
    def format_docs(retrieved_docs):
        # context = '\n'.join(docs.page_content for docs in retrieved_docs)
        for individual_chunk in retrieved_docs:
            context_with_metadata.append({'metadata':individual_chunk.metadata , 'page_content':individual_chunk.page_content})
        return context_with_metadata

    final_context = format_docs(context)

    main_chain = answer_prompt | model | parser

    result = main_chain.invoke({'context':final_context , 'question':user_input})

    st.write(result)