package dev.langchain4j.store.embedding.elasticsearch;

import co.elastic.clients.elasticsearch.ElasticsearchClient;
import co.elastic.clients.elasticsearch._types.ElasticsearchException;
import co.elastic.clients.elasticsearch.core.SearchResponse;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.rag.content.Content;
import dev.langchain4j.rag.content.ContentMetadata;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Supplier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Represents an <a href="https://www.elastic.co/">Elasticsearch</a> index as a text store
 * using full text search.
 *
 * @see <a href="https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-match-query">match query</a>
 */
public class ElasticsearchConfigurationFullText implements ElasticsearchConfiguration {
    private static final Logger log = LoggerFactory.getLogger(ElasticsearchConfigurationFullText.class);

    public static class Builder {
        public ElasticsearchConfigurationFullText build() {
            return new ElasticsearchConfigurationFullText();
        }
    }

    public static ElasticsearchConfigurationFullText.Builder builder() {
        return new ElasticsearchConfigurationFullText.Builder();
    }

    @Override
    public List<Content> retrieve(
            final ElasticsearchClient client,
            final String indexName,
            final Query query,
            final Supplier<EmbeddingSearchRequest> embeddingSupplier) {
        log.debug("retrieving with full text search([{}])", query.text());
        try {

            SearchResponse<Document> response = client.search(
                    s -> s.index(indexName)
                            .query(q -> q.match(m -> m.field(TEXT_FIELD).query(query.text()))),
                    Document.class);
            log.trace("found [{}] results", response);

            final List<Content> result = response.hits().hits().stream()
                    .map(hit -> Optional.ofNullable(hit.source())
                            .filter(document -> document.getText() != null)
                            .map(document -> {
                                Metadata metadata = new Metadata(document.getMetadata())
                                        .put(ContentMetadata.SCORE.name(), hit.score())
                                        .put(ContentMetadata.EMBEDDING_ID.name(), hit.id());

                                TextSegment segment = TextSegment.from(document.getText(), metadata);

                                return Content.from(
                                        segment,
                                        Map.of(
                                                ContentMetadata.SCORE, metadata.getDouble(ContentMetadata.SCORE.name()),
                                                ContentMetadata.EMBEDDING_ID,
                                                        metadata.getString(ContentMetadata.EMBEDDING_ID.name())));
                            }))
                    .flatMap(Optional::stream)
                    .toList();
            log.debug("Found [{}] relevant documents in Elasticsearch index [{}].", result.size(), indexName);
            return result;
        } catch (ElasticsearchException | IOException e) {
            throw new ElasticsearchRequestFailedException(e);
        }
    }
}
