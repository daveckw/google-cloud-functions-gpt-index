from llama_index import GPTSimpleVectorIndex
from typing import Any


class CustomGPTSimpleVectorIndex(GPTSimpleVectorIndex):
    def _delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        # Add your custom code here before the deletion

        # Check if the doc_id key exists in the doc_id_dict
        if doc_id in self._index_struct.doc_id_dict:
            document_ids = self._index_struct.doc_id_dict[doc_id]
            print("Deleting documents: \n", document_ids)
            for doc in document_ids:
                self._docstore.delete_document(doc)
        else:
            print(f"Document ID {doc_id} not found in the index")

        # Call the original _delete method from the parent class
        super()._delete(doc_id, **delete_kwargs)
