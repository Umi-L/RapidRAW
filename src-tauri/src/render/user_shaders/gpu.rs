//! GPU side of user shader passes: compile user fragment shaders into render pipelines
//! (cached per file version) and record the passes for one tile.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use wgpu::util::DeviceExt;

use super::parse::{ParsedShader, VERTEX_ENTRY};
use super::{FileStamp, load_shader, resolve_shader_file};
use crate::render::adjustment_utils::is_tool_active;
use crate::render::gpu::context::GpuContext;

/// Format of the intermediate textures the passes read and write.
pub const PASS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// `activeTools` key that bypasses the whole stack (the section's eye toggle).
pub const TOOL_ID: &str = "shaders";

#[derive(Clone)]
pub struct UserPass {
    pub path: PathBuf,
    pub stamp: FileStamp,
    pub shader: Arc<ParsedShader>,
    pub params: Vec<u8>,
    /// Slider values as stored in the adjustments, for `@iterations(param)`.
    pub values: Option<serde_json::Value>,
}

/// The enabled, successfully parsed passes of one image, in order.
#[derive(Clone, Default)]
pub struct UserPassStack {
    pub passes: Vec<UserPass>,
    pub is_preview: bool,
}

impl UserPassStack {
    /// Reads `shaderPasses` from the adjustments JSON. Disabled passes, missing files and
    /// files that fail to parse are skipped; they never fail the render.
    pub fn from_adjustments(adjustments: &serde_json::Value, is_preview: bool) -> Self {
        let mut stack = Self {
            passes: Vec::new(),
            is_preview,
        };
        if !is_tool_active(adjustments, TOOL_ID) {
            return stack;
        }
        let Some(list) = adjustments.get("shaderPasses").and_then(|v| v.as_array()) else {
            return stack;
        };
        let mask_layers = mask_layer_ids(adjustments);
        for entry in list {
            if entry.get("enabled").and_then(|v| v.as_bool()) == Some(false) {
                continue;
            }
            let Some(file) = entry.get("file").and_then(|v| v.as_str()) else {
                continue;
            };
            let Some(path) = resolve_shader_file(file) else {
                continue;
            };
            match load_shader(&path) {
                Ok((stamp, Ok(shader))) => stack.passes.push(UserPass {
                    params: shader.pack_params(entry.get("params"), &mask_layers),
                    values: entry.get("params").cloned(),
                    path,
                    stamp,
                    shader,
                }),
                Ok((_, Err(error))) | Err(error) => {
                    log::debug!("Skipping shader pass {}: {}", file, error)
                }
            }
        }
        stack
    }

    pub fn is_empty(&self) -> bool {
        self.passes.is_empty()
    }
}

/// Ids of the masks that get a GPU layer, in layer order. Must match the filter used by
/// `get_all_adjustments_from_json` and mask generation: visible masks with sub-masks.
fn mask_layer_ids(adjustments: &serde_json::Value) -> Vec<String> {
    adjustments
        .get("masks")
        .and_then(|m| m.as_array())
        .map(|masks| {
            masks
                .iter()
                .filter(|m| {
                    m.get("visible").and_then(|v| v.as_bool()).unwrap_or(false)
                        && m.get("subMasks")
                            .and_then(|s| s.as_array())
                            .is_some_and(|s| !s.is_empty())
                })
                .take(crate::render::uniforms::MAX_MASKS)
                .map(|m| {
                    m.get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string()
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Mirrors `struct RRUniforms` in prelude.wgsl.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct RRUniforms {
    image_size: [f32; 2],
    tile_origin: [f32; 2],
    tile_size: [f32; 2],
    scale: f32,
    hdr_headroom: f32,
    pass_index: u32,
    pass_count: u32,
    is_preview: u32,
    is_raw: u32,
    iteration: u32,
    iteration_count: u32,
    mask_count: u32,
    _pad: u32,
}

struct Shared {
    bgl: wgpu::BindGroupLayout,
    layout: wgpu::PipelineLayout,
    sampler: wgpu::Sampler,
}

/// Per-device pipeline state, stored on `GpuPipelines`.
#[derive(Default)]
pub struct UserPassPipelines {
    shared: OnceLock<Option<Shared>>,
    /// `None` marks a file version that failed to compile, so it isn't retried every frame.
    pipelines: Mutex<HashMap<(PathBuf, FileStamp), Option<wgpu::RenderPipeline>>>,
    pub encode_8bit: OnceLock<wgpu::ComputePipeline>,
    pub encode_high_precision: OnceLock<wgpu::ComputePipeline>,
}

/// One pass ready to record: its pipeline, params buffer and what decides its iterations.
pub struct PreparedPass {
    pipeline: wgpu::RenderPipeline,
    params: wgpu::Buffer,
    shader: Arc<ParsedShader>,
    values: Option<serde_json::Value>,
}

impl UserPassPipelines {
    fn shared(&self, context: &GpuContext) -> Option<&Shared> {
        self.shared
            .get_or_init(|| {
                let usages = context
                    .adapter
                    .get_texture_format_features(PASS_FORMAT)
                    .allowed_usages;
                let needed = wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING;
                if !usages.contains(needed) {
                    log::warn!(
                        "Shader passes are disabled: this GPU backend can't render to {:?}",
                        PASS_FORMAT
                    );
                    return None;
                }
                let device = &context.device;
                let fragment = wgpu::ShaderStages::FRAGMENT;
                let uniform = |binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: fragment,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                };
                let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("User Pass BGL"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: fragment,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: fragment,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        uniform(2),
                        uniform(3),
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: fragment,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                                view_dimension: wgpu::TextureViewDimension::D2Array,
                                multisampled: false,
                            },
                            count: None,
                        },
                    ],
                });
                let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("User Pass Layout"),
                    bind_group_layouts: &[Some(&bgl)],
                    immediate_size: 0,
                });
                let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("User Pass Sampler"),
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    ..Default::default()
                });
                Some(Shared {
                    bgl,
                    layout,
                    sampler,
                })
            })
            .as_ref()
    }

    fn pipeline(
        &self,
        context: &GpuContext,
        shared: &Shared,
        pass: &UserPass,
    ) -> Option<wgpu::RenderPipeline> {
        let key = (pass.path.clone(), pass.stamp);
        let mut cache = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(cached) = cache.get(&key) {
            return cached.clone();
        }
        // Drop pipelines for older versions of the same file.
        cache.retain(|(path, _), _| *path != pass.path);

        let device = &context.device;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("User Shader Pass"),
            source: wgpu::ShaderSource::Wgsl(pass.shader.source.as_str().into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("User Shader Pass Pipeline"),
            layout: Some(&shared.layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some(VERTEX_ENTRY),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some(&pass.shader.fragment_entry),
                targets: &[Some(wgpu::ColorTargetState {
                    format: PASS_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let result = match pollster::block_on(scope.pop()) {
            None => Some(pipeline),
            Some(error) => {
                log::warn!(
                    "Shader pass {} failed to compile: {}",
                    pass.path.display(),
                    error
                );
                None
            }
        };
        cache.insert(key, result.clone());
        result
    }

    /// Compiles (or fetches) every pass's pipeline and uploads its params. Passes that fail
    /// to compile are dropped; an empty result means "render without user passes".
    pub fn prepare(&self, context: &GpuContext, stack: &UserPassStack) -> Vec<PreparedPass> {
        if stack.is_empty() {
            return Vec::new();
        }
        let Some(shared) = self.shared(context) else {
            return Vec::new();
        };
        stack
            .passes
            .iter()
            .filter_map(|pass| {
                let pipeline = self.pipeline(context, shared, pass)?;
                let params = context
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("User Pass Params"),
                        contents: &pass.params,
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
                Some(PreparedPass {
                    pipeline,
                    params,
                    shader: Arc::clone(&pass.shader),
                    values: pass.values.clone(),
                })
            })
            .collect()
    }

    /// Records every prepared pass for one tile, ping-ponging between the two textures.
    /// A pass with `@iterations` is drawn that many times in a row. `textures[0]` holds
    /// the main shader's output on entry. Returns the index of the texture holding the
    /// final result.
    pub fn record(
        &self,
        context: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        passes: &[PreparedPass],
        textures: &PassTextures,
        masks: &wgpu::TextureView,
        tile: TileInfo,
    ) -> usize {
        let Some(shared) = self.shared(context) else {
            return 0;
        };
        let device = &context.device;
        let stride = (std::mem::size_of::<RRUniforms>() as u64)
            .next_multiple_of(context.limits.min_uniform_buffer_offset_alignment as u64);
        let mut current = 0;
        for (index, pass) in passes.iter().enumerate() {
            let iterations = pass
                .shader
                .iteration_count(pass.values.as_ref(), tile.scale);

            // One uniform block per iteration, each at an aligned offset of one buffer.
            let mut contents = vec![0u8; (stride * iterations as u64) as usize];
            for iteration in 0..iterations {
                let uniforms = RRUniforms {
                    image_size: [tile.image_size.0 as f32, tile.image_size.1 as f32],
                    tile_origin: [tile.origin.0 as f32, tile.origin.1 as f32],
                    tile_size: [tile.size.0 as f32, tile.size.1 as f32],
                    scale: tile.scale,
                    hdr_headroom: tile.hdr_headroom,
                    pass_index: index as u32,
                    pass_count: passes.len() as u32,
                    is_preview: tile.is_preview as u32,
                    is_raw: tile.is_raw as u32,
                    iteration,
                    iteration_count: iterations,
                    mask_count: tile.mask_count,
                    _pad: 0,
                };
                let at = (stride * iteration as u64) as usize;
                let bytes = bytemuck::bytes_of(&uniforms);
                contents[at..at + bytes.len()].copy_from_slice(bytes);
            }
            let rr = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("User Pass RR Uniforms"),
                contents: &contents,
                usage: wgpu::BufferUsages::UNIFORM,
            });

            for iteration in 0..iterations {
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("User Pass Bind Group"),
                    layout: &shared.bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&textures.views[current]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&shared.sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &rr,
                                offset: stride * iteration as u64,
                                size: wgpu::BufferSize::new(
                                    std::mem::size_of::<RRUniforms>() as u64
                                ),
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: pass.params.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(masks),
                        },
                    ],
                });
                let target = 1 - current;
                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("User Shader Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &textures.views[target],
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    rpass.set_viewport(0.0, 0.0, tile.size.0 as f32, tile.size.1 as f32, 0.0, 1.0);
                    rpass.set_scissor_rect(0, 0, tile.size.0, tile.size.1);
                    rpass.set_pipeline(&pass.pipeline);
                    rpass.set_bind_group(0, &bind_group, &[]);
                    rpass.draw(0..3, 0..1);
                }
                current = target;
            }
        }
        current
    }
}

#[derive(Clone, Copy)]
pub struct TileInfo {
    pub image_size: (u32, u32),
    pub origin: (u32, u32),
    pub size: (u32, u32),
    pub scale: f32,
    pub hdr_headroom: f32,
    pub is_preview: bool,
    pub is_raw: bool,
    /// Number of mask layers bound; `rr_mask` returns 0 for indices at or above this.
    pub mask_count: u32,
}

/// Two tile-sized float textures: the main shader writes `views[0]` as storage, then
/// passes ping-pong between them as render targets.
pub struct PassTextures {
    pub views: [wgpu::TextureView; 2],
}

impl PassTextures {
    pub fn new(device: &wgpu::Device, size: wgpu::Extent3d) -> Self {
        let make = |label| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: PASS_FORMAT,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        Self {
            views: [make("User Pass Texture A"), make("User Pass Texture B")],
        }
    }
}

#[cfg(test)]
mod tests {
    //! Runs real user passes on a real GPU device, end to end through `prepare` and
    //! `record`. Skips (passes) when the machine has no usable adapter.

    use super::*;
    use half::f16;

    const SIZE: u32 = 64;

    fn context() -> Option<GpuContext> {
        // Pick the adapter the way the app does (context.rs): primary backends, fastest GPU.
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
        if std::env::var("WGPU_BACKEND").is_err() {
            descriptor.backends = wgpu::Backends::PRIMARY;
        }
        let instance = wgpu::Instance::new(descriptor);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .ok()?;
        let info = adapter.get_info();
        eprintln!("test GPU: {} ({:?})", info.name, info.backend);
        let mut features = wgpu::Features::empty();
        if adapter
            .features()
            .contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES)
        {
            features |= wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        }
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("user shader test"),
            required_features: features,
            required_limits: limits.clone(),
            ..Default::default()
        }))
        .ok()?;
        Some(GpuContext {
            instance,
            backend: adapter.get_info().backend,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
            limits,
            display: Default::default(),
            viewer_display: Default::default(),
            pipelines: Default::default(),
            compute: Default::default(),
        })
    }

    fn texture(
        context: &GpuContext,
        format: wgpu::TextureFormat,
        layers: u32,
        data: &[u8],
    ) -> wgpu::TextureView {
        let texture = context.device.create_texture_with_data(
            &context.queue,
            &wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: SIZE,
                    height: SIZE,
                    depth_or_array_layers: layers,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            data,
        );
        texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(if layers > 1 {
                wgpu::TextureViewDimension::D2Array
            } else {
                wgpu::TextureViewDimension::D2
            }),
            ..Default::default()
        })
    }

    fn read_red(context: &GpuContext, view: &wgpu::TextureView) -> Vec<f32> {
        let device = &context.device;
        let row = SIZE * 8;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (row * SIZE) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            view.texture().as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(SIZE),
                },
            },
            wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
        );
        context.queue.submit(Some(encoder.finish()));
        buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let data = buffer.slice(..).get_mapped_range();
        bytemuck::cast_slice::<u8, f16>(&data)
            .chunks(4)
            .map(|px| px[0].to_f32())
            .collect()
    }

    #[test]
    fn probe_tmp_real_adjustments() {
        use crate::render::gpu::processor::{GpuProcessor, RenderOutputPrecision, RenderRequest, RenderedPixels};
        let context = context().unwrap();
        let appdata = std::env::var("APPDATA").unwrap();
        let _ = super::super::SHADERS_DIR.set(std::path::PathBuf::from(format!("{appdata}/io.github.CyberTimon.RapidRAW/shaders")));
        let home = std::env::var("USERPROFILE").unwrap();
        let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{home}/OneDrive/Pictures/Walk Oct 29 w mom/DSC02857.ARW.rrdata")).unwrap()).unwrap();
        let mut js = meta["adjustments"].clone();
        js["shaderPasses"][0]["params"]["amount"] = serde_json::json!(1.0);
        js["shaderPasses"][0]["params"]["tint"] = serde_json::json!([1.0, 0.0, 0.0]);
        let (w, h) = (1280u32, 850u32);
        let processor = GpuProcessor::new(context.clone(), 2304, 1536).unwrap();
        let halves: Vec<f16> = (0..w * h * 4).map(|_| f16::from_f32(0.2)).collect();
        let slot = crate::render::gpu::compute::upload_image(&context, (w, h), &halves);
        let guided = processor.build_guided_coeffs(&slot.view, w, h, 1);
        let entry = crate::app::state::GpuImageEntry { id: 1, texture: slot.texture, texture_view: slot.view, gf_coeffs_view: guided.coeffs, tone_base_view: guided.tone_base, haze_view: guided.haze, airlight_view: guided.airlight, is_raw: 1, width: w, height: h, transform_key: 0, source_identity: 0 };
        let masks = vec![image::GrayImage::from_pixel(w, h, image::Luma([255u8]))];
        let mask_view = processor.mask_view(&masks, w, h, None);
        for (round, with_passes) in [false, true].into_iter().enumerate() {
            let _ = round;
            let stack = if with_passes { UserPassStack::from_adjustments(&js, true) } else { UserPassStack::default() };
            eprintln!("PROBE passes in stack: {}", stack.passes.len());
            let adjustments = crate::render::uniforms::get_all_adjustments_from_json(&js, true, crate::decode::as_shot::AsShot::reference(), Default::default());
            eprintln!("PROBE mask_count uniform: {}", adjustments.mask_count);
            let request = RenderRequest { adjustments, mask_bitmaps: &masks, mask_cache_key: None, lut: None, roi: None, hsl_probe: false, motion_blur: Default::default(), user_passes: stack };
            let scope = context.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let (pixels, ..) = processor.run(&entry, &mask_view, request, true, RenderOutputPrecision::EightBit).unwrap();
            eprintln!("PROBE validation: {:?}", pollster::block_on(scope.pop()).map(|e| e.to_string()));
            let _ = pixels;
            let display = processor.display_textures(wgpu::TextureFormat::Rgba8Unorm);
            let device = &context.device;
            let buffer = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (5120 * 1536) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.copy_texture_to_buffer(display.working_texture.as_image_copy(), wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(5120), rows_per_image: Some(1536) } }, wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 });
            context.queue.submit(Some(encoder.finish()));
            buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let p = buffer.slice(..).get_mapped_range();
            eprintln!("PROBE display corner {:?} centre {:?}", &p[0..3], &p[(425 * 5120 + 640 * 4) as usize..][..3]);
        }
    }

    /// The full processor path: main shader in user-pass mode, the pass, then
    /// `user_encode`, with a mask texture bound as in a real edit.
    #[test]
    fn processor_runs_user_passes() {
        use crate::render::gpu::processor::{
            GpuProcessor, RenderOutputPrecision, RenderRequest, RenderedPixels,
        };
        let Some(context) = context() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("red.wgsl");
        std::fs::write(
            &path,
            "@fragment
fn main(in: RRInput) -> @location(0) vec4f {
    return vec4f(1.0, 0.0, 0.0, 1.0);
}
",
        )
        .unwrap();
        let (stamp, parsed) = load_shader(&path).unwrap();
        let shader = parsed.unwrap();
        let stack = UserPassStack {
            passes: vec![UserPass {
                params: shader.pack_params(None, &[]),
                values: None,
                path,
                stamp,
                shader,
            }],
            is_preview: true,
        };

        let processor = GpuProcessor::new(context.clone(), SIZE, SIZE).unwrap();
        let halves: Vec<f16> = (0..SIZE * SIZE * 4).map(|_| f16::from_f32(0.5)).collect();
        let slot = crate::render::gpu::compute::upload_image(&context, (SIZE, SIZE), &halves);
        let guided = processor.build_guided_coeffs(&slot.view, SIZE, SIZE, 0);
        let entry = crate::app::state::GpuImageEntry {
            id: 1,
            texture: slot.texture,
            texture_view: slot.view,
            gf_coeffs_view: guided.coeffs,
            tone_base_view: guided.tone_base,
            haze_view: guided.haze,
            airlight_view: guided.airlight,
            is_raw: 0,
            width: SIZE,
            height: SIZE,
            transform_key: 0,
            source_identity: 0,
        };
        let masks = vec![image::GrayImage::from_pixel(
            SIZE,
            SIZE,
            image::Luma([255u8]),
        )];
        let mask_view = processor.mask_view(&masks, SIZE, SIZE, None);

        let render = |user_passes: UserPassStack| {
            let adjustments = crate::render::uniforms::get_all_adjustments_from_json(
                &serde_json::json!({}),
                false,
                crate::decode::as_shot::AsShot::reference(),
                Default::default(),
            );
            let request = RenderRequest {
                adjustments,
                mask_bitmaps: &masks,
                mask_cache_key: None,
                lut: None,
                roi: None,
                hsl_probe: false,
                motion_blur: Default::default(),
                user_passes,
            };
            let scope = context
                .device
                .push_error_scope(wgpu::ErrorFilter::Validation);
            let (pixels, ..) = processor
                .run(
                    &entry,
                    &mask_view,
                    request,
                    false,
                    RenderOutputPrecision::EightBit,
                )
                .unwrap();
            if let Some(error) = pollster::block_on(scope.pop()) {
                panic!("wgpu validation error: {error}");
            }
            match pixels {
                RenderedPixels::U8(p) => [p[0], p[1], p[2]],
                RenderedPixels::U16(_) => unreachable!(),
            }
        };

        let plain = render(UserPassStack::default());
        let red = render(stack);
        assert_ne!(plain, red, "the pass must change the output");
        assert!(
            red[0] > 240 && red[1] < 15 && red[2] < 15,
            "expected red, got {red:?}"
        );
    }

    #[test]
    fn masked_iterated_pass_renders_on_gpu() {
        let Some(context) = context() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let source = "//! @iterations(3)\nstruct Params {\n    region: i32, // @mask\n    gain: f32, // @default(2)\n}\n@fragment\nfn main(in: RRInput) -> @location(0) vec4f {\n    let c = rr_color(in);\n    let m = rr_mask(params.region, vec2i(rr_px(in)));\n    return vec4f(c.rgb * mix(1.0, params.gain, m), c.a);\n}\n";
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.wgsl");
        std::fs::write(&path, source).unwrap();
        let (stamp, parsed) = load_shader(&path).unwrap();
        let shader = parsed.unwrap();
        let values = serde_json::json!({ "region": "m1" });
        let stack = UserPassStack {
            passes: vec![UserPass {
                params: shader.pack_params(Some(&values), &["m1".to_string()]),
                values: Some(values),
                path,
                stamp,
                shader,
            }],
            is_preview: true,
        };

        let device = &context.device;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let pipelines = UserPassPipelines::default();
        let prepared = pipelines.prepare(&context, &stack);
        assert_eq!(prepared.len(), 1, "the pass should compile");

        let textures = PassTextures::new(
            device,
            wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
        );
        let start: Vec<f16> = (0..SIZE * SIZE * 4).map(|_| f16::from_f32(0.1)).collect();
        context.queue.write_texture(
            textures.views[0].texture().as_image_copy(),
            bytemuck::cast_slice(&start),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(SIZE * 8),
                rows_per_image: Some(SIZE),
            },
            wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
        );
        // Layer 0 covers the left half; layer 1 is empty.
        let mut mask = vec![0u8; (SIZE * SIZE * 2) as usize];
        for y in 0..SIZE {
            for x in 0..SIZE / 2 {
                mask[(y * SIZE + x) as usize] = 255;
            }
        }
        let masks = texture(&context, wgpu::TextureFormat::R8Unorm, 2, &mask);

        let mut encoder = device.create_command_encoder(&Default::default());
        let last = pipelines.record(
            &context,
            &mut encoder,
            &prepared,
            &textures,
            &masks,
            TileInfo {
                image_size: (SIZE, SIZE),
                origin: (0, 0),
                size: (SIZE, SIZE),
                scale: 1.0,
                hdr_headroom: 1.0,
                is_preview: true,
                is_raw: false,
                mask_count: 1,
            },
        );
        context.queue.submit(Some(encoder.finish()));
        if let Some(error) = pollster::block_on(scope.pop()) {
            panic!("wgpu validation error: {error}");
        }

        let red = read_red(&context, &textures.views[last]);
        let inside = red[(10 * SIZE + 5) as usize];
        let outside = red[(10 * SIZE + 50) as usize];
        assert!(
            (inside - 0.8).abs() < 0.01,
            "inside mask: 0.1 doubled 3 times, got {inside}"
        );
        assert!(
            (outside - 0.1).abs() < 0.01,
            "outside mask stays 0.1, got {outside}"
        );
    }
}
