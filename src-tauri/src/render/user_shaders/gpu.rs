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
                    params: shader.pack_params(entry.get("params")),
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

/// One pass ready to record: its pipeline plus its params buffer.
pub struct PreparedPass {
    pipeline: wgpu::RenderPipeline,
    params: wgpu::Buffer,
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
                Some(PreparedPass { pipeline, params })
            })
            .collect()
    }

    /// Records every prepared pass for one tile, ping-ponging between the two textures.
    /// `textures[0]` holds the main shader's output on entry. Returns the index of the
    /// texture holding the final result.
    pub fn record(
        &self,
        context: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        passes: &[PreparedPass],
        textures: &PassTextures,
        tile: TileInfo,
    ) -> usize {
        let Some(shared) = self.shared(context) else {
            return 0;
        };
        let device = &context.device;
        let mut current = 0;
        for (index, pass) in passes.iter().enumerate() {
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
            };
            let rr = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("User Pass RR Uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
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
                        resource: rr.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: pass.params.as_entire_binding(),
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
                        | wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        Self {
            views: [make("User Pass Texture A"), make("User Pass Texture B")],
        }
    }
}
