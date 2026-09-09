-- film_docs holds exactly what the embedder is given, so a vector can always be
-- traced back to the text that produced it.

create table film_docs (
  tmdb_id              integer primary key references films(tmdb_id) on delete cascade,
  doc_template_version text not null,
  doc_kind             text not null check (doc_kind in ('full','no_overview','minimal')),
  title_text           text not null,
  people_text          text not null,
  keyword_text         text not null,
  overview_text        text not null,
  full_text            text not null,      -- exactly what the embedder receives
  overview_offset      integer not null,   -- where overview_text starts in full_text, -1 if absent
  doc_sha              text not null,
  built_at             text not null
) strict;
create index film_docs_sha  on film_docs(doc_sha);
create index film_docs_kind on film_docs(doc_kind);

-- External content, not contentless. A contentless fts5 table cannot DELETE at all, so a
-- renormalize leaves the old tokens indexed forever and the film keeps matching its old text.
-- Tokenizer is unicode61 only. Porter stems proper nouns, and film titles are proper nouns.
create virtual table films_fts using fts5(
  title_text, people_text, keyword_text, overview_text,
  content = 'film_docs',
  content_rowid = 'tmdb_id',
  tokenize = 'unicode61 remove_diacritics 2',
  prefix = '2 3'
);

create trigger film_docs_ai after insert on film_docs begin
  insert into films_fts(rowid, title_text, people_text, keyword_text, overview_text)
  values (new.tmdb_id, new.title_text, new.people_text, new.keyword_text, new.overview_text);
end;

create trigger film_docs_ad after delete on film_docs begin
  insert into films_fts(films_fts, rowid, title_text, people_text, keyword_text, overview_text)
  values ('delete', old.tmdb_id, old.title_text, old.people_text,
          old.keyword_text, old.overview_text);
end;

create trigger film_docs_au after update on film_docs begin
  insert into films_fts(films_fts, rowid, title_text, people_text, keyword_text, overview_text)
  values ('delete', old.tmdb_id, old.title_text, old.people_text,
          old.keyword_text, old.overview_text);
  insert into films_fts(rowid, title_text, people_text, keyword_text, overview_text)
  values (new.tmdb_id, new.title_text, new.people_text, new.keyword_text, new.overview_text);
end;
