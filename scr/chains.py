from langchain.chains import ConversationalRetrievalChain

def create_qa_chain(llm, retriever, memory, prompt):
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        return_generated_question=False,
        verbose=True
    )
