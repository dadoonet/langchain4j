package dev.langchain4j.store.embedding.elasticsearch;

import co.elastic.clients.elasticsearch.ElasticsearchClient;
import co.elastic.clients.elasticsearch._types.ElasticsearchException;
import co.elastic.clients.elasticsearch._types.KnnQuery;
import co.elastic.clients.elasticsearch.core.SearchResponse;
import co.elastic.clients.elasticsearch.core.search.SourceConfig;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.rag.content.Content;
import dev.langchain4j.rag.content.ContentMetadata;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Represents an <a href="https://www.elastic.co/">Elasticsearch</a> index as an embedding store
 * using the approximate kNN query implementation.
 *
 * @see <a href="https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-knn-query.html#knn-query-top-level-parameters">kNN query</a>
 */
public class ElasticsearchConfigurationKnn implements ElasticsearchConfiguration {
    private static final Logger log = LoggerFactory.getLogger(ElasticsearchConfigurationKnn.class);
    private final Integer numCandidates;
    private final boolean includeVectorResponse;

    public static class Builder {
        private Integer numCandidates;
        private boolean includeVectorResponse = false;

        public ElasticsearchConfigurationKnn build() {
            return new ElasticsearchConfigurationKnn(numCandidates, includeVectorResponse);
        }

        /**
         * The number of nearest neighbor candidates to consider per shard while doing knn search.
         * Cannot exceed 10,000. Increasing num_candidates tends to improve the accuracy of the final
         * results.
         *
         * @param numCandidates The number of nearest neighbor candidates to consider per shard
         * @return the builder instance
         */
        public Builder numCandidates(Integer numCandidates) {
            this.numCandidates = numCandidates;
            return this;
        }

        /**
         * Whether to include vector fields in the search response (from Elasticsearch 9.2).
         * Only useful for tests. Not recommended at all to use that in production.
         *
         * @param includeVectorResponse true to include vector fields, false otherwise
         * @return the builder instance
         */
        public Builder includeVectorResponse(boolean includeVectorResponse) {
            this.includeVectorResponse = includeVectorResponse;
            return this;
        }
    }

    public static ElasticsearchConfigurationKnn.Builder builder() {
        return new Builder();
    }

    private ElasticsearchConfigurationKnn(final Integer numCandidates, final boolean includeVectorResponse) {
        this.numCandidates = numCandidates;
        this.includeVectorResponse = includeVectorResponse;
    }

    @Override
    public SearchResponse<Document> vectorSearch(
            ElasticsearchClient client, String indexName, EmbeddingSearchRequest embeddingSearchRequest)
            throws ElasticsearchException, IOException {
        KnnQuery.Builder krb = new KnnQuery.Builder()
                .field(VECTOR_FIELD)
                .queryVector(embeddingSearchRequest.queryEmbedding().vectorAsList());

        if (embeddingSearchRequest.filter() != null) {
            krb.filter(ElasticsearchMetadataFilterMapper.map(embeddingSearchRequest.filter()));
        }

        if (numCandidates != null) {
            krb.numCandidates(numCandidates);
        }

        KnnQuery knn = krb.build();

        log.trace("Searching for embeddings in index [{}] with query [{}].", indexName, knn);

        return client.search(
                s -> s.source(sr -> {
                            if (includeVectorResponse) {
                                return sr.filter(f -> f.excludeVectors(false));
                            }
                            return new SourceConfig.Builder().filter(f -> f);
                        })
                        .index(indexName)
                        .size(embeddingSearchRequest.maxResults())
                        .query(q -> q.knn(knn))
                        .minScore(embeddingSearchRequest.minScore()),
                Document.class);
    }

    @Override
    public List<Content> retrieve(
            final ElasticsearchClient client,
            final String indexName,
            final Query query,
            final Supplier<EmbeddingSearchRequest> embeddingSupplier) {
        log.debug("retrieving with knn search([...{}...])", query.text());
        try {
            final EmbeddingSearchRequest request = embeddingSupplier.get();
            SearchResponse<Document> response = vectorSearch(client, indexName, request);
            log.trace("found [{}] results", response);

            List<Content> result = response.hits().hits().stream()
                    .filter(hit -> hit.source() != null)
                    .filter(hit -> hit.score()
                            > request.minScore()) // I don't think this is needed as ES is filtering it anyway
                    .map(hit -> {
                        Document document = hit.source();
                        TextSegment textSegment = document.getText() == null
                                ? null
                                : TextSegment.from(document.getText(), new Metadata(document.getMetadata()));

                        return Content.from(
                                textSegment,
                                Map.of(
                                        ContentMetadata.SCORE, hit.score(),
                                        ContentMetadata.EMBEDDING_ID, hit.id()));
                    })
                    .toList();

            log.debug("Found [{}] relevant documents in Elasticsearch index [{}].", result.size(), indexName);
            return result;
        } catch (ElasticsearchException e) {
            String message = String.valueOf(e.getLocalizedMessage());
            if (includeVectorResponse && message.contains("Unknown key for a VALUE_BOOLEAN in [exclude_vectors]")) {
                log.warn(
                        "Property [includeVectorResponse] is not needed for elasticsearch server versions previous to 9.2, remove it to fix the exception.");
            }
            throw new ElasticsearchRequestFailedException(e);
        } catch (IOException e) {
            throw new ElasticsearchRequestFailedException(e);
        }
    }
}
