package dev.langchain4j.store.embedding.elasticsearch;

import co.elastic.clients.elasticsearch.ElasticsearchClient;
import co.elastic.clients.elasticsearch._types.ElasticsearchException;
import co.elastic.clients.elasticsearch.core.SearchResponse;
import dev.langchain4j.rag.content.Content;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import java.io.IOException;
import java.util.List;
import java.util.function.Supplier;

public interface ElasticsearchConfiguration {
    String VECTOR_FIELD = "vector";
    String TEXT_FIELD = "text";

    /**
     * Used for vector search
     *
     * @param client                 The Elasticsearch client
     * @param indexName              The index name
     * @param embeddingSearchRequest The embedding search request
     * @return The search response
     * @throws ElasticsearchException if an error occurs during the search
     * @throws IOException            if an I/O error occurs
     */
    default SearchResponse<Document> vectorSearch(
            ElasticsearchClient client, String indexName, EmbeddingSearchRequest embeddingSearchRequest)
            throws ElasticsearchException, IOException {
        throw new UnsupportedOperationException(
                this.getClass().getSimpleName() + " configuration does not support vector search");
    }

    /**
     * Retrieve content depending on the configuration
     *
     * @param client                The Elasticsearch client
     * @param indexName             The index name
     * @param query                 The query to retrieve content for
     * @param embeddingSupplier     The embedding search request supplier
     * @return The list of content retrieved
     */
    default List<Content> retrieve(
            ElasticsearchClient client,
            String indexName,
            final Query query,
            final Supplier<EmbeddingSearchRequest> embeddingSupplier) {
        throw new UnsupportedOperationException(
                this.getClass().getSimpleName() + " configuration does not support retrieval");
    }
}
