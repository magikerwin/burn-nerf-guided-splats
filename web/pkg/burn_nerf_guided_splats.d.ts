/* tslint:disable */
/* eslint-disable */

export class WasmTrainingSession {
    free(): void;
    [Symbol.dispose](): void;
    get_gaussian_render(): Uint8Array;
    get_nerf_importance_map(): Uint8Array;
    get_nerf_render(): Uint8Array;
    constructor(width: number, height: number, num_gaussians: number, target_rgb: Uint8Array);
    seed_from_nerf(): void;
    step_gaussian(lr: number): number;
    step_nerf(lr: number): number;
}

export function init_panic_hook(): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wasmtrainingsession_free: (a: number, b: number) => void;
    readonly wasmtrainingsession_get_gaussian_render: (a: number) => [number, number];
    readonly wasmtrainingsession_get_nerf_importance_map: (a: number) => [number, number];
    readonly wasmtrainingsession_get_nerf_render: (a: number) => [number, number];
    readonly wasmtrainingsession_new: (a: number, b: number, c: number, d: number, e: number) => number;
    readonly wasmtrainingsession_seed_from_nerf: (a: number) => void;
    readonly wasmtrainingsession_step_gaussian: (a: number, b: number) => number;
    readonly wasmtrainingsession_step_nerf: (a: number, b: number) => number;
    readonly init_panic_hook: () => void;
    readonly wasm_bindgen__convert__closures_____invoke__ha47ab804cced2a29: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen__convert__closures_____invoke__h053ce51d358c81af: (a: number, b: number, c: any) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_destroy_closure: (a: number, b: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
