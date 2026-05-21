from typing import Callable, Iterable, List, Optional, Union  # List used by return type hints

from .corpus import CorpusRecord, SearchResult, read_records_jsonl, write_records_jsonl

DeviceSpec = Union[str, List[str]]


class LateOnIndex:
    def __init__(
        self,
        index_folder: str,
        index_name: str = "officeqa_lateon",
        records_file: Optional[str] = None,
        model_name: str = "lightonai/LateOn",
        batch_size: int = 32,
        search_batch_size: Optional[int] = None,
        n_full_scores: Optional[int] = None,
        n_ivf_probe: Optional[int] = None,
        device: Optional[DeviceSpec] = None,
    ):
        indexes, models, retrieve = _import_pylate()
        plaid_kwargs = _plaid_kwargs(
            search_batch_size=search_batch_size,
            n_full_scores=n_full_scores,
            n_ivf_probe=n_ivf_probe,
            device=device,
        )
        self.model = models.ColBERT(model_name_or_path=model_name, device=_model_device(device))
        self.index = indexes.PLAID(
            index_folder=index_folder,
            index_name=index_name,
            override=False,
            **plaid_kwargs,
        )
        self.retriever = retrieve.ColBERT(index=self.index)
        self.records = {record.record_id: record for record in read_records_jsonl(records_file)} if records_file else {}
        self.batch_size = batch_size

    @classmethod
    def build(
        cls,
        records: Iterable[CorpusRecord],
        index_folder: str,
        index_name: str = "officeqa_lateon",
        records_file: Optional[str] = None,
        model_name: str = "lightonai/LateOn",
        batch_size: int = 32,
        search_batch_size: Optional[int] = None,
        n_full_scores: Optional[int] = None,
        n_ivf_probe: Optional[int] = None,
        device: Optional[DeviceSpec] = None,
        override: bool = True,
        nbits: int = 4,
        kmeans_niters: int = 4,
    ) -> "LateOnIndex":
        indexes, models, _ = _import_pylate()
        records = list(records)
        if records_file:
            write_records_jsonl(records, records_file)

        plaid_kwargs = _plaid_kwargs(
            search_batch_size=search_batch_size,
            n_full_scores=n_full_scores,
            n_ivf_probe=n_ivf_probe,
            device=device,
        )
        model = models.ColBERT(model_name_or_path=model_name, device=_model_device(device))
        index = indexes.PLAID(
            index_folder=index_folder,
            index_name=index_name,
            override=override,
            nbits=nbits,
            kmeans_niters=kmeans_niters,
            **plaid_kwargs,
        )

        embeddings = model.encode(
            [_lateon_text(record) for record in records],
            batch_size=batch_size,
            is_query=False,
            show_progress_bar=True,
        )
        index.add_documents(
            documents_ids=[record.record_id for record in records],
            documents_embeddings=embeddings,
        )

        return cls(
            index_folder=index_folder,
            index_name=index_name,
            records_file=records_file,
            model_name=model_name,
            batch_size=batch_size,
            search_batch_size=search_batch_size,
            n_full_scores=n_full_scores,
            n_ivf_probe=n_ivf_probe,
            device=device,
        )

    def search(
        self,
        query: str,
        k: int = 100,
        filter_fn: Optional[Callable[[CorpusRecord], bool]] = None,
        subset: Optional[Iterable[str]] = None,
    ) -> List[SearchResult]:
        subset_ids = _dedupe_subset(subset)
        if subset_ids is not None and not subset_ids:
            return []

        embeddings = self.model.encode(
            [query],
            batch_size=1,
            is_query=True,
            show_progress_bar=False,
        )
        raw = self._retrieve(embeddings, k, subset=subset_ids)
        results = []
        for rank, item in enumerate(raw, start=1):
            record_id = str(_field(item, "id"))
            record = self.records.get(record_id)
            if record is None or (filter_fn and not filter_fn(record)):
                continue
            score = float(_field(item, "score") or 0.0)
            results.append(
                SearchResult(
                    record=record,
                    score=score,
                    rank=rank,
                    channel="lateon",
                    score_breakdown={"lateon": score},
                )
            )
        return results

    def _retrieve(self, embeddings, k: int, subset: Optional[List[str]] = None):
        try:
            results = self.retriever.retrieve(queries_embeddings=embeddings, k=k, subset=subset)
        except AttributeError:
            results = self.index(embeddings, k=k, subset=subset)
        return results[0] if results else []


def _import_pylate():
    try:
        from pylate import indexes, models, retrieve
    except ImportError as exc:
        raise RuntimeError(
            "LateOn requires PyLate. Install it in a Python 3.11+ environment with: "
            "pip install -U pylate"
        ) from exc
    return indexes, models, retrieve


def _lateon_text(record) -> str:
    """Return the text to encode for a corpus record.

    LateOn has a 300-token document limit, so we prioritise actual content
    over metadata.  Title is prepended (short) so the model sees context,
    then the raw text fills the rest of the budget.
    """
    parts = [record.title, record.text] if record.title else [record.text]
    return "\n".join(p for p in parts if p)


def _plaid_kwargs(
    search_batch_size: Optional[int],
    n_full_scores: Optional[int],
    n_ivf_probe: Optional[int],
    device: Optional[DeviceSpec],
) -> dict:
    kwargs = {}
    if search_batch_size is not None:
        kwargs["batch_size"] = search_batch_size
    if n_full_scores is not None:
        kwargs["n_full_scores"] = n_full_scores
    if n_ivf_probe is not None:
        kwargs["n_ivf_probe"] = n_ivf_probe
    if device is not None:
        kwargs["device"] = device
    return kwargs


def _model_device(device: Optional[DeviceSpec]) -> Optional[str]:
    if isinstance(device, list):
        return device[0] if device else None
    return device


def _dedupe_subset(subset: Optional[Iterable[str]]) -> Optional[List[str]]:
    if subset is None:
        return None

    seen = set()
    deduped = []
    for record_id in subset:
        record_id = str(record_id)
        if record_id not in seen:
            deduped.append(record_id)
            seen.add(record_id)
    return deduped


def _field(item, name: str):
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)
