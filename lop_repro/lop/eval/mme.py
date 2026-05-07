from __future__ import annotations

from dataclasses import dataclass

# MME 官方定义的感知（Perception）和认知（Cognition）任务分类
PERCEPTION_TASKS = (
    "existence",
    "count",
    "position",
    "color",
    "posters",
    "celebrity",
    "scene",
    "landmark",
    "artwork",
    "OCR",
)
COGNITION_TASKS = (
    "commonsense_reasoning",
    "numerical_calculation",
    "text_translation",
    "code_reasoning",
)

# 单个 MME 子任务的评分结果
@dataclass(frozen=True)
class MmeTaskScore:
    task: str            # 任务名（如 artwork, code_reasoning）
    questions: int       # 该任务的题目总数
    image_groups: int    # 该任务的图片组数（每个图片组包含 2 道题）
    accuracy: float      # 题目级正确率（答对题数 / 总题数）
    accuracy_plus: float # 图片组级全对率（全部答对的图片组数 / 总图片组数）
    score: float         # 该任务最终得分 = (accuracy + accuracy_plus) × 100

# MME 完整评分结果
@dataclass(frozen=True)
class MmeScore:
    perception: float              # 所有感知任务得分之和
    cognition: float               # 所有认知任务得分之和
    total: float                   # perception + cognition
    tasks: tuple[MmeTaskScore, ...] # 每个子任务的详细评分

# 对 MME 数据集答案进行官方评分。
#
# MME 评分公式（两个维度）：
#   1. accuracy: 按题算正确率（每道题独立判对错）
#   2. accuracy_plus: 按图片组算全对率（同一图片的 2 道题都答对才算全对）
#   3. score = (accuracy + accuracy_plus) × 100
#
# 最终：
#   - MME-P (Perception): 10 个感知任务得分之和，满分 2000
#   - MME-R (Cognition):  4 个认知任务得分之和，满分 800
#   - MME Total: MME-P + MME-R, 满分 2800
def score_mme(samples: list[dict], predicted_labels: list[str]) -> MmeScore:
    if len(samples) != len(predicted_labels):
        raise ValueError(f"samples {len(samples)} != predictions {len(predicted_labels)}")
    grouped: dict[str, list[tuple[dict, str]]] = {}
    for sample, prediction in zip(samples, predicted_labels):
        if sample["dataset"] != "mme":
            raise ValueError(f"score_mme only accepts mme samples, got {sample['dataset']}")
        category = sample["fields"]["category"]
        grouped.setdefault(category, []).append((sample, prediction))

    task_scores = []
    for category in sorted(grouped):
        rows = grouped[category]
        correct = [int(_normalize_answer(sample["fields"]["answer"]) == _normalize_answer(prediction)) for sample, prediction in rows]

        # 按图片分组：同一张图片的 2 道题归到一组
        groups: dict[str, list[int]] = {}
        for (sample, _), is_correct in zip(rows, correct):
            image_key = sample["images"][0]["path"]
            groups.setdefault(image_key, []).append(is_correct)

        accuracy = sum(correct) / len(correct)
        # accuracy_plus: 图片组内全部答对才算 1，否则算 0
        accuracy_plus = sum(int(all(values)) for values in groups.values()) / len(groups)
        task_scores.append(
            MmeTaskScore(
                task=category,
                questions=len(rows),
                image_groups=len(groups),
                accuracy=accuracy,
                accuracy_plus=accuracy_plus,
                score=(accuracy + accuracy_plus) * 100.0,
            )
        )

    perception = sum(score.score for score in task_scores if score.task in PERCEPTION_TASKS)
    cognition = sum(score.score for score in task_scores if score.task in COGNITION_TASKS)
    return MmeScore(perception=perception, cognition=cognition, total=perception + cognition, tasks=tuple(task_scores))

# 归一化 yes/no 答案：取首单词小写，前缀匹配 "yes"/"no"，其余原样返回
def _normalize_answer(value: str) -> str:
    text = value.strip().lower()
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    return text
