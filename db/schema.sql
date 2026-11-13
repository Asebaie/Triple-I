
\restrict os49hWhtU7OW4yaEgg202ME7GihxJEfRTNSt6XI3gSiWocY2fmvL4YSNSN4cPVn


SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;



CREATE SCHEMA legal_assistant;



CREATE FUNCTION legal_assistant.prevent_used_model_dimension_change() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    IF NEW.dimensions <> OLD.dimensions
       AND EXISTS (
           SELECT 1
           FROM legal_assistant.chunk_embeddings
           WHERE embedding_model_id = OLD.id
       )
    THEN
        RAISE EXCEPTION
            'Cannot change dimensions: embedding model % is already in use',
            OLD.id;
    END IF;

    RETURN NEW;
END;
$$;




CREATE FUNCTION legal_assistant.validate_embedding_dimensions() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
DECLARE
    expected_dimensions INTEGER;
    actual_dimensions INTEGER;
BEGIN
    SELECT dimensions
    INTO expected_dimensions
    FROM legal_assistant.embedding_models
    WHERE id = NEW.embedding_model_id;

    actual_dimensions := cardinality(NEW.embedding);

    IF actual_dimensions <> expected_dimensions THEN
        RAISE EXCEPTION
            'Embedding dimension mismatch: expected %, received %',
            expected_dimensions,
            actual_dimensions;
    END IF;

    RETURN NEW;
END;
$$;


SET default_tablespace = '';

SET default_table_access_method = heap;


CREATE TABLE legal_assistant.chunk_embeddings (
    chunk_id bigint NOT NULL,
    embedding_model_id bigint NOT NULL,
    embedding real[] NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT chunk_embeddings_not_empty CHECK ((cardinality(embedding) > 0))
);




CREATE TABLE legal_assistant.chunks (
    id bigint NOT NULL,
    version_id bigint NOT NULL,
    chunk_index integer NOT NULL,
    content text NOT NULL,
    article_ref text,
    section_ref text,
    page_start integer,
    page_end integer,
    source_locator jsonb,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT chunks_chunk_index_check CHECK ((chunk_index >= 0)),
    CONSTRAINT chunks_content_not_blank CHECK ((length(btrim(content)) > 0)),
    CONSTRAINT chunks_page_range_valid CHECK (((page_end IS NULL) OR ((page_start IS NOT NULL) AND (page_end >= page_start)))),
    CONSTRAINT chunks_valid_pages CHECK (((page_start IS NULL) OR (page_start > 0)))
);




ALTER TABLE legal_assistant.chunks ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME legal_assistant.chunks_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);





CREATE TABLE legal_assistant.document_versions (
    id bigint NOT NULL,
    document_id bigint NOT NULL,
    version_seq integer NOT NULL,
    version_label text,
    effective_from date,
    effective_to date,
    end_kind text DEFAULT 'unknown'::text NOT NULL,
    approval_date date,
    source_filename text NOT NULL,
    file_sha256 character(64),
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT document_versions_end_kind_check CHECK ((end_kind = ANY (ARRAY['dated'::text, 'open'::text, 'unknown'::text]))),
    CONSTRAINT document_versions_end_kind_consistency CHECK ((((end_kind = 'dated'::text) AND (effective_to IS NOT NULL)) OR ((end_kind = ANY (ARRAY['open'::text, 'unknown'::text])) AND (effective_to IS NULL)))),
    CONSTRAINT document_versions_filename_not_blank CHECK ((length(btrim(source_filename)) > 0)),
    CONSTRAINT document_versions_valid_dates CHECK (((effective_to IS NULL) OR (effective_from IS NULL) OR (effective_to >= effective_from))),
    CONSTRAINT document_versions_version_seq_check CHECK ((version_seq > 0))
);


CREATE VIEW legal_assistant.conflicting_version_periods AS
 SELECT v1.document_id,
    v1.id AS version_1_id,
    v2.id AS version_2_id,
    v1.version_label AS version_1_label,
    v2.version_label AS version_2_label,
    v1.effective_from AS version_1_start,
    v1.effective_to AS version_1_end,
    v2.effective_from AS version_2_start,
    v2.effective_to AS version_2_end
   FROM (legal_assistant.document_versions v1
     JOIN legal_assistant.document_versions v2 ON (((v1.document_id = v2.document_id) AND (v1.id < v2.id))))
  WHERE ((v1.end_kind = ANY (ARRAY['dated'::text, 'open'::text])) AND (v2.end_kind = ANY (ARRAY['dated'::text, 'open'::text])) AND (v1.effective_from IS NOT NULL) AND (v2.effective_from IS NOT NULL) AND (v1.effective_from <= COALESCE(v2.effective_to, 'infinity'::date)) AND (v2.effective_from <= COALESCE(v1.effective_to, 'infinity'::date)));



CREATE TABLE legal_assistant.document_relations (
    id bigint NOT NULL,
    source_document_id bigint NOT NULL,
    target_document_id bigint NOT NULL,
    relation_type text NOT NULL,
    target_ref text,
    source_chunk_id bigint,
    basis text NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT document_relations_basis_check CHECK ((basis = ANY (ARRAY['source_metadata'::text, 'document_text'::text, 'manual'::text]))),
    CONSTRAINT document_relations_no_self_reference CHECK ((source_document_id <> target_document_id)),
    CONSTRAINT document_relations_relation_type_check CHECK ((relation_type = ANY (ARRAY['amends'::text, 'supersedes'::text, 'other'::text]))),
    CONSTRAINT document_relations_target_ref_not_blank CHECK (((target_ref IS NULL) OR (length(btrim(target_ref)) > 0)))
);


ALTER TABLE legal_assistant.document_relations ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME legal_assistant.document_relations_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


ALTER TABLE legal_assistant.document_versions ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME legal_assistant.document_versions_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);




CREATE TABLE legal_assistant.documents (
    id bigint NOT NULL,
    external_id text,
    title text NOT NULL,
    doc_type text NOT NULL,
    doc_number text,
    issuing_authority text,
    document_date date,
    history_policy text DEFAULT 'unknown'::text NOT NULL,
    metadata jsonb DEFAULT '{}'::jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT documents_doc_type_check CHECK ((doc_type = ANY (ARRAY['law'::text, 'regulation'::text, 'contract'::text, 'supplementary_agreement'::text, 'other'::text]))),
    CONSTRAINT documents_history_policy_check CHECK ((history_policy = ANY (ARRAY['current_only'::text, 'full_history'::text, 'unknown'::text]))),
    CONSTRAINT documents_title_not_blank CHECK ((length(btrim(title)) > 0))
);




ALTER TABLE legal_assistant.documents ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME legal_assistant.documents_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


CREATE VIEW legal_assistant.documents_without_versions AS
 SELECT d.id,
    d.title,
    d.doc_type
   FROM (legal_assistant.documents d
     LEFT JOIN legal_assistant.document_versions v ON ((v.document_id = d.id)))
  WHERE (v.id IS NULL);




CREATE VIEW legal_assistant.duplicate_file_candidates AS
 SELECT document_versions.file_sha256,
    count(*) AS occurrences,
    array_agg(document_versions.id) AS version_ids
   FROM legal_assistant.document_versions
  WHERE (document_versions.file_sha256 IS NOT NULL)
  GROUP BY document_versions.file_sha256
 HAVING (count(*) > 1);




CREATE TABLE legal_assistant.embedding_models (
    id bigint NOT NULL,
    name text NOT NULL,
    revision text NOT NULL,
    dimensions integer NOT NULL,
    is_active boolean DEFAULT false NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT embedding_models_dimensions_check CHECK ((dimensions > 0))
);



ALTER TABLE legal_assistant.embedding_models ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME legal_assistant.embedding_models_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);



ALTER TABLE ONLY legal_assistant.chunk_embeddings
    ADD CONSTRAINT chunk_embeddings_primary_key PRIMARY KEY (chunk_id, embedding_model_id);



ALTER TABLE ONLY legal_assistant.chunks
    ADD CONSTRAINT chunks_pkey PRIMARY KEY (id);



ALTER TABLE ONLY legal_assistant.chunks
    ADD CONSTRAINT chunks_version_index_unique UNIQUE (version_id, chunk_index);



ALTER TABLE ONLY legal_assistant.document_relations
    ADD CONSTRAINT document_relations_pkey PRIMARY KEY (id);



ALTER TABLE ONLY legal_assistant.document_versions
    ADD CONSTRAINT document_versions_hash_unique UNIQUE (document_id, file_sha256);



ALTER TABLE ONLY legal_assistant.document_versions
    ADD CONSTRAINT document_versions_pkey PRIMARY KEY (id);



ALTER TABLE ONLY legal_assistant.document_versions
    ADD CONSTRAINT document_versions_sequence_unique UNIQUE (document_id, version_seq);



ALTER TABLE ONLY legal_assistant.documents
    ADD CONSTRAINT documents_external_id_key UNIQUE (external_id);



ALTER TABLE ONLY legal_assistant.documents
    ADD CONSTRAINT documents_pkey PRIMARY KEY (id);



ALTER TABLE ONLY legal_assistant.embedding_models
    ADD CONSTRAINT embedding_models_name_revision_unique UNIQUE (name, revision);



ALTER TABLE ONLY legal_assistant.embedding_models
    ADD CONSTRAINT embedding_models_pkey PRIMARY KEY (id);



CREATE INDEX idx_chunk_embeddings_model ON legal_assistant.chunk_embeddings USING btree (embedding_model_id);



CREATE INDEX idx_chunks_article_ref ON legal_assistant.chunks USING btree (article_ref) WHERE (article_ref IS NOT NULL);



CREATE INDEX idx_chunks_version_id ON legal_assistant.chunks USING btree (version_id);



CREATE INDEX idx_documents_doc_number ON legal_assistant.documents USING btree (doc_number);



CREATE INDEX idx_documents_doc_type ON legal_assistant.documents USING btree (doc_type);



CREATE INDEX idx_relations_source_document ON legal_assistant.document_relations USING btree (source_document_id, relation_type);



CREATE INDEX idx_relations_target_document ON legal_assistant.document_relations USING btree (target_document_id, relation_type);



CREATE INDEX idx_versions_document_id ON legal_assistant.document_versions USING btree (document_id);



CREATE INDEX idx_versions_effective_dates ON legal_assistant.document_versions USING btree (effective_from, effective_to);



CREATE UNIQUE INDEX uq_document_relations_with_target_ref ON legal_assistant.document_relations USING btree (source_document_id, target_document_id, relation_type, target_ref) WHERE (target_ref IS NOT NULL);



CREATE UNIQUE INDEX uq_document_relations_without_target_ref ON legal_assistant.document_relations USING btree (source_document_id, target_document_id, relation_type) WHERE (target_ref IS NULL);



CREATE UNIQUE INDEX uq_document_versions_one_open ON legal_assistant.document_versions USING btree (document_id) WHERE (end_kind = 'open'::text);



CREATE UNIQUE INDEX uq_embedding_models_one_active ON legal_assistant.embedding_models USING btree (is_active) WHERE (is_active = true);



CREATE TRIGGER trg_prevent_used_model_dimension_change BEFORE UPDATE OF dimensions ON legal_assistant.embedding_models FOR EACH ROW EXECUTE FUNCTION legal_assistant.prevent_used_model_dimension_change();



CREATE TRIGGER trg_validate_embedding_dimensions BEFORE INSERT OR UPDATE ON legal_assistant.chunk_embeddings FOR EACH ROW EXECUTE FUNCTION legal_assistant.validate_embedding_dimensions();



ALTER TABLE ONLY legal_assistant.chunk_embeddings
    ADD CONSTRAINT chunk_embeddings_chunk_id_fkey FOREIGN KEY (chunk_id) REFERENCES legal_assistant.chunks(id) ON DELETE CASCADE;



ALTER TABLE ONLY legal_assistant.chunk_embeddings
    ADD CONSTRAINT chunk_embeddings_embedding_model_id_fkey FOREIGN KEY (embedding_model_id) REFERENCES legal_assistant.embedding_models(id) ON DELETE RESTRICT;



ALTER TABLE ONLY legal_assistant.chunks
    ADD CONSTRAINT chunks_version_id_fkey FOREIGN KEY (version_id) REFERENCES legal_assistant.document_versions(id) ON DELETE CASCADE;



ALTER TABLE ONLY legal_assistant.document_relations
    ADD CONSTRAINT document_relations_source_chunk_id_fkey FOREIGN KEY (source_chunk_id) REFERENCES legal_assistant.chunks(id) ON DELETE SET NULL;



ALTER TABLE ONLY legal_assistant.document_relations
    ADD CONSTRAINT document_relations_source_document_id_fkey FOREIGN KEY (source_document_id) REFERENCES legal_assistant.documents(id) ON DELETE CASCADE;



ALTER TABLE ONLY legal_assistant.document_relations
    ADD CONSTRAINT document_relations_target_document_id_fkey FOREIGN KEY (target_document_id) REFERENCES legal_assistant.documents(id) ON DELETE CASCADE;



ALTER TABLE ONLY legal_assistant.document_versions
    ADD CONSTRAINT document_versions_document_id_fkey FOREIGN KEY (document_id) REFERENCES legal_assistant.documents(id) ON DELETE CASCADE;



\unrestrict os49hWhtU7OW4yaEgg202ME7GihxJEfRTNSt6XI3gSiWocY2fmvL4YSNSN4cPVn

