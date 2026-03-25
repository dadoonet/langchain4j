package dev.langchain4j.store.embedding.elasticsearch;

import co.elastic.clients.elasticsearch.ElasticsearchClient;
import co.elastic.clients.elasticsearch._types.ElasticsearchException;
import co.elastic.clients.elasticsearch._types.KnnRetriever;
import co.elastic.clients.elasticsearch._types.RRFRetrieverEntry;
import co.elastic.clients.elasticsearch._types.query_dsl.MatchQuery;
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
 * using rff to combine a kNN query and a full text search.
 *
 * @see <a href="https://www.elastic.co/search-labs/tutorials/search-tutorial/vector-search/hybrid-search">hybrid search</a>
 * <br>
 * Running hybrid search requires an elasticsearch paid license.
 *
 * @see <a href="https://www.elastic.co/subscriptions">subscriptions</a>
 */
public class ElasticsearchConfigurationHybrid implements ElasticsearchConfiguration {
    private static final Logger log = LoggerFactory.getLogger(ElasticsearchConfigurationHybrid.class);
    private final Integer numCandidates;
    private final boolean includeVectorResponse;

    public static class Builder {
        private Integer numCandidates;
        private boolean includeVectorResponse = false;

        public ElasticsearchConfigurationHybrid build() {
            return new ElasticsearchConfigurationHybrid(numCandidates, includeVectorResponse);
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

    public static ElasticsearchConfigurationHybrid.Builder builder() {
        return new Builder();
    }

    private ElasticsearchConfigurationHybrid(final Integer numCandidates, final boolean includeVectorResponse) {
        this.numCandidates = numCandidates;
        this.includeVectorResponse = includeVectorResponse;
    }

    @Override
    public List<Content> retrieve(
            final ElasticsearchClient client,
            final String indexName,
            final Query query,
            final Supplier<EmbeddingSearchRequest> embeddingSupplier) {
        log.debug("retrieving with knn search([...{}...])", query.text());
        try {
            EmbeddingSearchRequest request = embeddingSupplier.get();

            // Building KNN part of the hybrid query
            KnnRetriever.Builder krb = new KnnRetriever.Builder()
                    .field(VECTOR_FIELD)
                    .queryVector(request.queryEmbedding().vectorAsList());

            if (request.filter() != null) {
                krb.filter(ElasticsearchMetadataFilterMapper.map(request.filter()));
            }

            // k and numCandidates are required in KnnRetriever, calculating default values similarly to how
            // elasticsearch
            // calculates them for KnnQuery
            if (numCandidates != null) {
                krb.numCandidates(numCandidates);
                krb.k(Math.min(numCandidates, request.maxResults()));
            } else {
                krb.numCandidates(request.maxResults());
                krb.k(request.maxResults());
            }

            KnnRetriever knn = krb.build();

            // Building full text part of the hybrid query
            MatchQuery matchQuery = new MatchQuery.Builder()
                    .field(TEXT_FIELD)
                    .query(query.text())
                    .build();

            log.trace(
                    "Searching for embeddings in index [{}] with hybrid query [{}], [{}].", indexName, knn, matchQuery);

            SearchResponse<Document> response = client.search(
                    s -> s.source(sr -> {
                                if (includeVectorResponse) {
                                    return sr.filter(f -> f.excludeVectors(false));
                                }
                                return new SourceConfig.Builder().filter(f -> f);
                            })
                            .index(indexName)
                            .retriever(r -> r.rrf(rf -> rf.retrievers(List.of(
                                    RRFRetrieverEntry.of(
                                            rre -> rre.retriever(rt -> rt.standard(st -> st.query(matchQuery)))),
                                    RRFRetrieverEntry.of(rre -> rre.retriever(rt -> rt.knn(knn)))))))
                            .size(request.maxResults())
                            .minScore(request.minScore()),
                    Document.class);

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
