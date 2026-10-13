"""Typed experiment configuration contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crypto_steering.records import ProviderOutcome
from crypto_steering.selection import Selection


@dataclass(frozen=True)
class ExperimentPaths:
    dataset_root: Path
    output_root: Path


@dataclass(frozen=True)
class ConfigReferences:
    model: Path
    tasks: Path
    steering_prompts: Path
    validation: Path
    steering: Path
    pca: Path
    pca_interpretation_probes: Path
    layer_probes: Path


@dataclass(frozen=True)
class ComputeConfig:
    device: str
    cpu_threads: int
    cpu_service_threads: int
    overlap_cpu_gpu: bool
    max_concurrent_cpu_audits: int


@dataclass(frozen=True)
class ExtractionConfig:
    shard_size: int
    include_embedding_layer: bool
    storage_dtype: str
    order: str


@dataclass(frozen=True)
class GeneralSubsetConfig:
    subset_id: str
    cells: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class VectorConfig:
    structures: tuple[str, ...]
    source_levels: tuple[str, ...]
    summary_methods: tuple[str, ...]
    general_methods: tuple[str, ...]
    general_subsets: tuple[GeneralSubsetConfig, ...]


@dataclass(frozen=True)
class CoreGeometryConfig:
    normalization_epsilon: float
    rsa_similarity: str
    cluster_counts: tuple[int, ...]
    cluster_random_seed: int
    kmeans_n_init: int
    kmeans_max_iter: int
    kmeans_tolerance: float


@dataclass(frozen=True)
class PlotConfig:
    signed_cosine_limits: tuple[float, float]
    absolute_cosine_limits: tuple[float, float]
    norm_ratio_log2_limits: tuple[float, float]
    rsa_cosine_limits: tuple[float, float]
    write_companion_tables: bool


@dataclass(frozen=True)
class DiagnosticConfig:
    histogram_bins: int
    nearest_neighbors: int
    outliers_per_tail: int
    neighbor_block_rows: int


@dataclass(frozen=True)
class ResamplingProfileConfig:
    profile_id: str
    bootstrap_samples: int
    permutation_samples: int
    gap_reference_samples: int


@dataclass(frozen=True)
class BootstrapConfig:
    layers: str | tuple[int, ...]
    resampling_mode: str
    cluster_counts: tuple[int, ...]
    algorithms: tuple[str, ...]
    natural_k_min: int
    natural_k_max: int


@dataclass(frozen=True)
class AdvancedConfig:
    random_seed: int
    confidence_level: float
    subspace_rank: int
    normalize_hosvd_vectors: bool
    misuse_families: tuple[tuple[str, str], ...]
    bootstrap: BootstrapConfig
    default_resampling_profile: str
    resampling_profiles: tuple[ResamplingProfileConfig, ...]
    steering_outcomes: Path | None

    def misuse_family(self, misuse_id: str) -> str:
        try:
            return dict(self.misuse_families)[misuse_id]
        except KeyError as exc:
            raise KeyError(f"No semantic family is configured for {misuse_id}") from exc

    def profile(self, profile_id: str | None = None) -> ResamplingProfileConfig:
        selected = profile_id or self.default_resampling_profile
        for profile in self.resampling_profiles:
            if profile.profile_id == selected:
                return profile
        raise KeyError(f"Unknown resampling profile {selected!r}")


@dataclass(frozen=True)
class PcaResamplingProfileConfig:
    profile_id: str
    permutation_samples: int
    bootstrap_samples: int


@dataclass(frozen=True)
class PcaCrossLayerConfig:
    method: str
    reference_layer: str | int


@dataclass(frozen=True)
class PcaConfig:
    schema_version: int
    config_id: str
    source_levels: tuple[str, ...]
    input_modes: tuple[str, ...]
    component_count: int
    security_component_candidate_count: int
    resampling_seed: int
    orientation_epsilon: float
    projection_basis_scope: str
    scope_kinds: tuple[str, ...]
    multivariate_component_sets: tuple[int, ...]
    exact_sign_flip_max_pairs: int
    default_resampling_profile: str
    resampling_profiles: tuple[PcaResamplingProfileConfig, ...]
    cross_layer: PcaCrossLayerConfig

    def profile(self, profile_id: str | None = None) -> PcaResamplingProfileConfig:
        selected = profile_id or self.default_resampling_profile
        for profile in self.resampling_profiles:
            if profile.profile_id == selected:
                return profile
        raise KeyError(f"Unknown PCA resampling profile {selected!r}")


@dataclass(frozen=True)
class LayerProbeTorchConfig:
    dtype: str
    fold_batch_size: int
    history_size: int
    line_search: str


@dataclass(frozen=True)
class LayerProbeSelectionConfig:
    prompt_family: str
    readout: str
    metric: str
    protocol_weights: tuple[tuple[str, float], ...]
    exact_tie_break: str
    write_maximin_sensitivity: bool
    require_full_coordinate_coverage: bool
    require_all_transformer_layers: bool

    def protocol_weight(self, protocol: str) -> float:
        try:
            return dict(self.protocol_weights)[protocol]
        except KeyError as exc:
            raise KeyError(f"No layer-selection weight for {protocol!r}") from exc


@dataclass(frozen=True)
class LayerProbeConfig:
    schema_version: int
    config_id: str
    protocols: tuple[str, ...]
    backend: str
    regularization_c: float
    solver: str
    max_iterations: int
    tolerance: float
    standardize: bool
    random_seed: int
    sample_weighting: str
    torch: LayerProbeTorchConfig
    layer_selection: LayerProbeSelectionConfig


@dataclass(frozen=True)
class PcaInterpretationProbeConfig:
    schema_version: int
    config_id: str
    corpus_root: Path
    prompt_family: str
    readouts: tuple[str, ...]
    conditions: tuple[str, ...]
    dose_conditions: tuple[str, ...]
    component_count: int
    basis_scope: str
    source_levels: tuple[str, ...]
    input_modes: tuple[str, ...]
    security_composition: PcaSecurityCompositionConfig


@dataclass(frozen=True)
class PcaSecurityCompositionConfig:
    corpus_root: Path
    prompt_family: str
    readouts: tuple[str, ...]
    component_count: int
    basis_scope: str
    source_levels: tuple[str, ...]
    input_modes: tuple[str, ...]
    bit_order: tuple[str, ...]
    programs_per_language: int
    program_count: int
    preferred_anchor_layer: int
    default_centering: str
    default_view: str
    default_color: str
    default_reversed_components: tuple[int, ...]


@dataclass(frozen=True)
class PcaVisualizationConfig:
    default_layer: str | int
    source_level: str
    input_mode: str
    point_population: str
    components: tuple[int, ...]
    default_view: str
    presets: tuple[str, ...]


@dataclass(frozen=True)
class RawHiddenVisualizationConfig:
    dimensions: tuple[int, ...]
    max_points: int | None
    random_seed: int
    cluster_counts: tuple[int, ...]


@dataclass(frozen=True)
class HeatmapVisualizationConfig:
    layer: str | int
    score_percentile: float
    prompt_mode: str
    sample_count: int
    enabled_direction_modes: tuple[str, ...]
    initial_signed_percentile: float


@dataclass(frozen=True)
class VisualizationConfig:
    pca: PcaVisualizationConfig
    raw_hidden: RawHiddenVisualizationConfig
    heatmaps: HeatmapVisualizationConfig


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    experiment_id: str
    project_root: Path
    paths: ExperimentPaths
    configs: ConfigReferences
    selection: Selection
    supported_prompt_families: tuple[str, ...]
    supported_readouts: tuple[str, ...]
    compute: ComputeConfig
    extraction: ExtractionConfig
    vectors: VectorConfig
    core_geometry: CoreGeometryConfig
    plots: PlotConfig
    diagnostics: DiagnosticConfig
    advanced: AdvancedConfig
    visualization: VisualizationConfig


@dataclass(frozen=True)
class ModelGenerationConfig:
    max_new_tokens: int


@dataclass(frozen=True)
class ModelConfig:
    schema_version: int
    run_id: str
    model_id: str
    backend: str
    torch_dtype: str
    device_map: str
    trust_remote_code: bool
    batch_size: int
    prompt_format: str
    chat_template: str
    padding_side: str
    generation: ModelGenerationConfig


@dataclass(frozen=True)
class ReplayLanguageConfig:
    language: str
    opening: str
    source_contract: str
    dependency_contract: tuple[str, ...]
    fence_label: str
    source_placeholder: str


@dataclass(frozen=True)
class ReplayConfig:
    task_prefix: str
    common_instructions: tuple[str, ...]
    languages: tuple[ReplayLanguageConfig, ...]


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    public_objective: str
    completion_contract: str


@dataclass(frozen=True)
class TaskFamilyConfig:
    family_id: str
    functionality: str
    target_misuse_ids: tuple[str, ...]
    controlled_roles: tuple[str, ...]
    tasks: tuple[TaskSpec, ...]


@dataclass(frozen=True)
class TaskConfig:
    schema_version: int
    config_id: str
    replay: ReplayConfig
    families: tuple[TaskFamilyConfig, ...]

    def task(self, task_id: str) -> TaskSpec:
        for family in self.families:
            for task in family.tasks:
                if task.task_id == task_id:
                    return task
        raise KeyError(task_id)

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(task.task_id for family in self.families for task in family.tasks)


@dataclass(frozen=True)
class SteeringPrompt:
    language: str
    prompt_id: str
    misuse_id: str
    prompt: str


@dataclass(frozen=True)
class SteeringPromptConfig:
    schema_version: int
    config_id: str
    expected_prompts_per_cell: int
    prompts: tuple[SteeringPrompt, ...]


@dataclass(frozen=True)
class ValidationProviderConfig:
    provider_id: str
    capabilities: tuple[str, ...]

    @property
    def supported(self) -> bool:
        return bool(self.capabilities)

    def unsupported_outcome(self) -> ProviderOutcome:
        if self.supported:
            raise ValueError(
                f"Provider {self.provider_id!r} declares capabilities and is not unsupported"
            )
        return ProviderOutcome.unsupported(
            f"Validation provider {self.provider_id!r} has no implemented capabilities"
        )


@dataclass(frozen=True)
class RewriteRuleConfig:
    misuse_id: str
    secure: tuple[str, ...]
    misuse: tuple[str, ...]


@dataclass(frozen=True)
class ValidationPolicyConfig:
    label_minimum_iterations: int
    modern_guidance_iterations: int
    rewrite_rules: tuple[RewriteRuleConfig, ...]
    structures: tuple[str, ...]
    task_families: tuple[tuple[str, str], ...]

@dataclass(frozen=True)
class ValidationFrontendConfig:
    language: str
    values: tuple[tuple[str, Any], ...]

    @property
    def raw(self) -> dict[str, Any]:
        return dict(self.values)


@dataclass(frozen=True)
class ValidationExecutionConfig:
    workers: int
    analysis_timeout_seconds: int


@dataclass(frozen=True)
class ValidationDatasetAuditConfig:
    expected_programs: int
    expected_originals: int
    expected_rewrites: int
    samples_per_task: tuple[tuple[str, int], ...]
    task_ids: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True)
class ValidationProbeCorpusConfig:
    corpus_id: str
    root: Path
    program_count: int
    intended_label_assertions: int


@dataclass(frozen=True)
class ValidationProbeAuditConfig:
    corpora: tuple[ValidationProbeCorpusConfig, ...]

    def corpus(self, corpus_id: str) -> ValidationProbeCorpusConfig:
        for corpus in self.corpora:
            if corpus.corpus_id == corpus_id:
                return corpus
        raise KeyError(corpus_id)


@dataclass(frozen=True)
class ValidationConfig:
    schema_version: int
    config_id: str
    selected_provider: str
    providers: tuple[ValidationProviderConfig, ...]
    policy: ValidationPolicyConfig
    frontends: tuple[ValidationFrontendConfig, ...]
    execution: ValidationExecutionConfig
    dataset_audit: ValidationDatasetAuditConfig
    probe_audit: ValidationProbeAuditConfig

    def provider(self, provider_id: str) -> ValidationProviderConfig:
        for provider in self.providers:
            if provider.provider_id == provider_id:
                return provider
        raise KeyError(provider_id)

    def frontend(self, language: str) -> ValidationFrontendConfig:
        for frontend in self.frontends:
            if frontend.language == language:
                return frontend
        raise KeyError(language)


@dataclass(frozen=True)
class SteeringLayerConfig:
    default: str
    model_overrides: tuple[tuple[str, int], ...]

    def for_model(self, run_id: str) -> str | int:
        return dict(self.model_overrides).get(run_id, self.default)


@dataclass(frozen=True)
class SteeringVectorConfig:
    vector_id: str
    scope: str
    source_level: str
    method: str
    normalization: str
    component: int | str | None


@dataclass(frozen=True)
class SteeringApplicationConfig:
    application_id: str
    stage: str
    alpha_schedule: str
    vectors: tuple[str, ...]
    prompt_ids: str | tuple[str, ...]
    target_languages: str | tuple[str, ...]
    target_misuses: str | tuple[str, ...]


@dataclass(frozen=True)
class AlphaScheduleConfig:
    comparison_alphas: tuple[float, ...]
    security_pc_sweep_alphas: tuple[float, ...]
    heldout_alphas: tuple[float, ...]

    def for_schedule(self, schedule: str) -> tuple[float, ...]:
        return {
            "comparison": self.comparison_alphas,
            "security_pc_sweep": self.security_pc_sweep_alphas,
            "heldout": self.heldout_alphas,
        }[schedule]


@dataclass(frozen=True)
class InterventionConfig:
    application_site: str
    normalize_vector: bool
    coefficient_scale: str
    baseline: str


@dataclass(frozen=True)
class SteeringGenerationConfig:
    samples_per_prompt: int
    stage_base_seed: int
    do_sample: bool
    temperature: float
    top_p: float
    max_new_tokens: int


@dataclass(frozen=True)
class SteeringResamplingProfileConfig:
    profile_id: str
    bootstrap_samples: int


@dataclass(frozen=True)
class SteeringReportingConfig:
    confidence_level: float
    paired_bootstrap_mode: str
    default_resampling_profile: str
    resampling_profiles: tuple[SteeringResamplingProfileConfig, ...]
    aggregation_levels: tuple[str, ...]

    def profile(self, profile_id: str | None = None) -> SteeringResamplingProfileConfig:
        selected = profile_id or self.default_resampling_profile
        for profile in self.resampling_profiles:
            if profile.profile_id == selected:
                return profile
        raise KeyError(selected)


@dataclass(frozen=True)
class SteeringConfig:
    schema_version: int
    config_id: str
    prompt_family: str
    readout: str
    layers: SteeringLayerConfig
    vectors: tuple[SteeringVectorConfig, ...]
    applications: tuple[SteeringApplicationConfig, ...]
    prompt_task_overrides: tuple[tuple[str, str], ...]
    alphas: AlphaScheduleConfig
    intervention: InterventionConfig
    generation: SteeringGenerationConfig
    reporting: SteeringReportingConfig


@dataclass(frozen=True)
class ConfigBundle:
    experiment: ExperimentConfig
    model: ModelConfig
    tasks: TaskConfig
    steering_prompts: SteeringPromptConfig
    validation: ValidationConfig
    steering: SteeringConfig
    pca: PcaConfig
    pca_interpretation_probes: PcaInterpretationProbeConfig
    layer_probes: LayerProbeConfig


def resolve_device(requested: str, *, cuda_available: bool) -> str:
    """Resolve a configured device without importing a model framework."""

    if requested == "auto":
        return "cuda" if cuda_available else "cpu"
    if requested == "cuda":
        if not cuda_available:
            raise RuntimeError("CUDA was requested explicitly but is unavailable")
        return "cuda"
    if requested == "cpu":
        return "cpu"
    raise ValueError("device must be one of: auto, cuda, cpu")
