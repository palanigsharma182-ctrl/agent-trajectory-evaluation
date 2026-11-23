from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


SupportStatus = Literal["SUPPORTED", "PARTIALLY_SUPPORTED", "REFUTED", "UNKNOWN"]
RetrievalMode = Literal["paragraph", "unit", "dependency"]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Paragraph(Record):
    id: str
    section: str
    text: str = Field(min_length=1)


class Paper(Record):
    id: str
    title: str
    abstract: str
    source_url: str
    source_split: str
    paragraphs: list[Paragraph] = Field(min_length=1)


class SourceSpan(Record):
    paragraph_id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    quote: str = Field(min_length=1)


class Proposition(Record):
    id: str
    kind: Literal["METHOD", "SETTING", "RESULT", "COMPARISON", "INTERPRETATION"]
    statement: str = Field(min_length=1)
    paragraph_id: str
    quote: str = Field(min_length=1)

    def anchor(self, paragraph: Paragraph) -> SourceSpan:
        if paragraph.id != self.paragraph_id or self.quote not in paragraph.text:
            raise ValueError(f"命题 {self.id} 无法定位到段落 {self.paragraph_id}")
        start = paragraph.text.index(self.quote)
        return SourceSpan(paragraph_id=paragraph.id, start=start, end=start + len(self.quote), quote=self.quote)


class Relation(Record):
    source_id: str
    target_id: str
    kind: Literal["REQUIRES", "SUPPORTS", "DERIVED_FROM", "CONTRADICTS", "EQUIVALENT"]


class Experiment(Record):
    id: str
    setting_ids: list[str]
    result_ids: list[str]


class Assessment(Record):
    id: str
    claim_id: str
    status: SupportStatus
    proposition_ids: list[str]
    source_ids: list[str]
    reason: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    assessor: str = Field(min_length=1)
    revision: int = Field(default=1, ge=1)


class Task(Record):
    id: str
    question: str = Field(min_length=1)
    result_requirement: str
    attribution_requirement: str


class Demo(Record):
    result_paragraph_id: str
    ablation_paragraph_id: str
    result_query: str
    ablation_query: str
    result_answer: str
    attribution_answer: str
    result_proposition_ids: list[str] = Field(min_length=1)
    ablation_proposition_ids: list[str] = Field(min_length=1)


class Fixture(Record):
    paper: Paper
    task: Task
    propositions: list[Proposition] = Field(min_length=1)
    relations: list[Relation]
    experiment: Experiment
    demo: Demo

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        paragraphs = {item.id: item for item in self.paper.paragraphs}
        propositions = {item.id: item for item in self.propositions}
        if len(paragraphs) != len(self.paper.paragraphs) or len(propositions) != len(self.propositions):
            raise ValueError("段落 ID 或命题 ID 重复")
        for item in self.propositions:
            paragraph = paragraphs.get(item.paragraph_id)
            if paragraph is None:
                raise ValueError(f"命题 {item.id} 引用了不存在的段落")
            item.anchor(paragraph)
        for relation in self.relations:
            if relation.source_id not in propositions or relation.target_id not in propositions:
                raise ValueError("关系端点必须属于同一篇论文")
        ids = self.experiment.setting_ids + self.experiment.result_ids
        ids += self.demo.result_proposition_ids + self.demo.ablation_proposition_ids
        if not set(ids).issubset(propositions):
            raise ValueError("实验或演示引用了未入库的命题")
        if not {self.demo.result_paragraph_id, self.demo.ablation_paragraph_id}.issubset(paragraphs):
            raise ValueError("演示引用了未入库的段落")
        if self.demo.result_paragraph_id == self.demo.ablation_paragraph_id:
            raise ValueError("结果证据和待屏蔽的消融证据必须能分别读取")
        return self


class OfflineReview(Record):
    assessor: str
    assessments: list[Assessment] = Field(min_length=1)


class Search(Record):
    query: str = Field(min_length=1, max_length=400)


class Read(Record):
    paragraph_ids: list[str] = Field(min_length=1, max_length=3)


class Finish(Record):
    answer: str = Field(min_length=1, max_length=5000)
    evidence_ids: list[str]


class ModelAction(Record):
    name: Literal["search", "read", "finish"]
    arguments: dict
    action_reason: str = Field(min_length=1)
    source_ids: list[str] = Field(default_factory=list)


class RunConfig(Record):
    retrieval: RetrievalMode = "paragraph"
    frontier: bool = True
    max_tool_calls: int = Field(default=7, ge=2, le=12)
    max_context_chars: int = Field(default=42000, gt=0)


class DependencyGroup(Record):
    operator: Literal["ALL", "ANY"] = "ALL"
    required_ids: list[str] = Field(min_length=1)
