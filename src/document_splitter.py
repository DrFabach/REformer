"""
Document splitting utilities for medical text processing.

This module provides implementations of document splitters that divide
large documents into manageable chunks while preserving entity and relation annotations.
"""

from typing import Union, List, Optional
from medkit.core import Attribute, Operation
from medkit.core.text import (
    Entity,
    ModifiedSpan,
    Relation,
    Segment,
    Span,
    TextAnnotation,
    TextDocument,
    span_utils,
)
from medkit.text.postprocessing.alignment_utils import compute_nested_segments


class DocumentSplitterSlidingCharacter(Operation):
    """
    Split text documents into sliding windows of characters.
    
    This splitter preserves entity annotations and relations within each window.
    """

    def __init__(
        self,
        window_size: int,
        entity_labels: Optional[List[str]] = None,
        attr_labels: Optional[List[str]] = None,
        relation_labels: Optional[List[str]] = None,
        name: Optional[str] = None,
        uid: Optional[str] = None,
        overlap: bool = True
    ):
        """
        Initialize the document splitter.
        
        Parameters
        ----------
        window_size : int
            Number of characters to include in each window
        entity_labels : list[str], optional
            Labels of entities to include in the mini documents.
            If None, all entities will be included.
        attr_labels : list[str], optional
            Labels of attributes to include in the new annotations.
            If None, all attributes will be included.
        relation_labels : list[str], optional
            Labels of relations to include in the mini documents.
            If None, all relations will be included.
        name : str, optional
            Name describing the splitter
        uid : str, optional
            Identifier of the operation
        overlap : bool, default=True
            Whether to create overlapping windows. If True, the stride is set to
            half the window size. If False, a small overlap is still maintained
            to avoid cutting entities at window boundaries.
        """
        # Pass arguments to the parent class
        init_args = locals()
        init_args.pop("self")
        super().__init__(**init_args)

        self.window_size = window_size
        if overlap:
            self.stride = window_size // 2  # Set stride to half the window size
        else:
            self.stride = window_size - 20  # Small overlap to avoid cutting entities
        self.entity_labels = entity_labels
        self.attr_labels = attr_labels
        self.relation_labels = relation_labels

    def run(self, docs: list[TextDocument]) -> list[TextDocument]:
        """
        Split documents into mini documents using sliding windows.
        
        Parameters
        ----------
        docs: list of TextDocument
            List of text documents to split

        Returns
        -------
        list of TextDocument
            List of documents created from the sliding windows
        """
        window_docs = []

        for doc in docs:
            text = doc.text
            text_length = len(text)

            for window_start in range(0, text_length, self.stride):
                window_end = min(window_start + self.window_size, text_length)
                if window_start >= text_length:
                    break

                # Get entities in the window
                window_entities = self._get_entities_in_window(doc, window_start, window_end)

                # Get relations in the window
                window_relations = self._get_relations_in_window(doc, window_entities)

                # Create new document from window
                window_doc = self._create_window_doc(
                    window_start=window_start,
                    window_end=window_end,
                    entities=window_entities,
                    relations=window_relations,
                    doc_source=doc,
                )
                window_docs.append(window_doc)
        
        return window_docs

    def _get_entities_in_window(self, doc: TextDocument, window_start: int, window_end: int) -> list[Entity]:
        """Get entities that are fully contained within the window."""
        entities = (
            doc.anns.get_entities()
            if self.entity_labels is None
            else [ent for label in self.entity_labels for ent in doc.anns.get_entities(label=label)]
        )
        return [ent for ent in entities if self._entity_in_window(ent, window_start, window_end)]

    def _get_relations_in_window(self, doc: TextDocument, window_entities: list[Entity]) -> list[Relation]:
        """Get relations between entities in the window."""
        relations = (
            doc.anns.get_relations()
            if self.relation_labels is None
            else [rel for label in self.relation_labels for rel in doc.anns.get_relations(label=label)]
        )
        entities_uid = {ent.uid for ent in window_entities}
        return [
            relation
            for relation in relations
            if relation.source_id in entities_uid and relation.target_id in entities_uid
        ]

    def _create_window_doc(
        self,
        window_start: int,
        window_end: int,
        entities: list[Entity],
        relations: list[Relation],
        doc_source: TextDocument,
    ) -> TextDocument:
        """Create a new document from a window with its entities and relations."""
        metadata = doc_source.metadata.copy()
        metadata.update({
            "window_start": window_start,
            "window_end": window_end,
            "path_to_text": doc_source.metadata.get("path_to_text", "")
        })

        window_doc = TextDocument(text=doc_source.text[window_start:window_end], metadata=metadata)

        if self._prov_tracer is not None:
            self._prov_tracer.add_prov(window_doc, self.description, source_data_items=[doc_source])

        uid_mapping = {}
        
        # Add entities to the new document
        for ent in entities:
            relocated_ent = self._relocate_entity(ent, window_start)
            uid_mapping[ent.uid] = relocated_ent.uid
            window_doc.anns.add(relocated_ent)

        # Add relations to the new document
        for rel in relations:
            relocated_rel = self._relocate_relation(rel, uid_mapping)
            window_doc.anns.add(relocated_rel)

        return window_doc

    def _relocate_entity(self, entity: Entity, window_start: int) -> Entity:
        """Adjust entity spans to the new document coordinates."""
        spans = []
        for span in entity.spans:
            if isinstance(span, Span):
                spans.append(Span(span.start - window_start, span.end - window_start))
            else:
                replaced_spans = [Span(sp.start - window_start, sp.end - window_start) for sp in span.replaced_spans]
                spans.append(ModifiedSpan(length=span.length, replaced_spans=replaced_spans))

        relocated_ent = Entity(
            text=entity.text,
            label=entity.label,
            spans=spans,
            metadata=entity.metadata.copy(),
        )

        if self._prov_tracer is not None:
            self._prov_tracer.add_prov(relocated_ent, self.description, source_data_items=[entity])

        for attr in self._filter_attrs_from_ann(entity):
            new_attr = attr.copy()
            relocated_ent.attrs.add(new_attr)
            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(new_attr, self.description, source_data_items=[attr])

        return relocated_ent

    def _relocate_relation(self, relation: Relation, uid_mapping: dict) -> Relation:
        """Create a new relation with updated entity references."""
        relocated_rel = Relation(
            label=relation.label,
            source_id=uid_mapping[relation.source_id],
            target_id=uid_mapping[relation.target_id],
            metadata=relation.metadata.copy(),
        )

        if self._prov_tracer is not None:
            self._prov_tracer.add_prov(relocated_rel, self.description, source_data_items=[relation])

        for attr in self._filter_attrs_from_ann(relation):
            new_attr = attr.copy()
            relocated_rel.attrs.add(new_attr)
            if self._prov_tracer is not None:
                self._prov_tracer.add_prov(new_attr, self.description, source_data_items=[attr])

        return relocated_rel

    def _filter_attrs_from_ann(self, ann: TextAnnotation) -> list[Attribute]:
        """Filter attributes from an annotation using 'attr_labels'."""
        return (
            ann.attrs.get()
            if self.attr_labels is None
            else [attr for label in self.attr_labels for attr in ann.attrs.get(label=label)]
        )

    def _entity_in_window(self, entity: Entity, window_start: int, window_end: int) -> bool:
        """Check if an entity is fully contained within a window."""
        span_i = span_utils.normalize_spans(entity.spans)
        entity_start = span_i[0].start
        entity_end = span_i[-1].end
        return entity_start >= window_start and entity_end <= window_end

# Helper function to use the document splitter with default settings
def split_docs_sliding_character(docs, window_size=500):
    """Split documents into windows using the sliding character method."""
    doc_splitter = DocumentSplitterSlidingCharacter(window_size=window_size)
    return doc_splitter.run(docs)
