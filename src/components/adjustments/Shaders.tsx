import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { useTranslation } from 'react-i18next';
import { AlertTriangle, ChevronDown, ChevronUp, FolderOpen, RefreshCw, X } from 'lucide-react';
import clsx from 'clsx';
import { v4 as uuidv4 } from 'uuid';
import Dropdown from '../ui/Dropdown';
import Slider from '../ui/Slider';
import Switch from '../ui/Switch';
import Text from '../ui/Text';
import VisibilityToggle, { HIDDEN_CONTENT_CLASS } from '../ui/VisibilityToggle';
import { TextColors, TextVariants } from '../../types/typography';
import { Invokes } from '../../types/invokes';
import { Adjustments, ShaderPass, areSectionPropsEqual } from '../../utils/adjustments';
import { useEditorStore } from '../../store/useEditorStore';

/** Mirrors `ParamSpec` in src-tauri/src/render/user_shaders/parse.rs. */
interface ShaderParamSpec {
  name: string;
  label: string;
  kind: 'float' | 'int' | 'uint';
  components: number;
  min: number;
  max: number;
  step: number;
  default: Array<number>;
  toggle: boolean;
  /** An `i32` that selects one of the image's masks; its value is the mask id. */
  mask: boolean;
  /** Labels for an integer field shown as a dropdown; its value is the chosen index. */
  options: Array<string>;
}

/** Mirrors `UserShaderInfo` in src-tauri/src/render/user_shaders/mod.rs. */
interface UserShaderInfo {
  file: string;
  name: string;
  description: string;
  params: Array<ShaderParamSpec>;
  error: string | null;
  revision: string;
}

const COMPONENT_NAMES = ['x', 'y', 'z', 'w'];

const defaultParams = (shader: UserShaderInfo): ShaderPass['params'] =>
  Object.fromEntries(
    shader.params.map((spec) => [
      spec.name,
      spec.mask ? '' : spec.components === 1 ? spec.default[0] : [...spec.default],
    ]),
  );

const paramValue = (pass: ShaderPass, spec: ShaderParamSpec, component: number): number => {
  const value = pass.params?.[spec.name];
  if (Array.isArray(value)) return value[component] ?? spec.default[component];
  if (typeof value === 'number' && spec.components === 1) return value;
  return spec.default[component];
};

const maskValue = (pass: ShaderPass, spec: ShaderParamSpec): string => {
  const value = pass.params?.[spec.name];
  return typeof value === 'string' ? value : '';
};

interface ShadersPanelProps {
  adjustments: Adjustments;
  setAdjustments(updater: (prev: Adjustments) => Adjustments): void;
  onDragStateChange?: (isDragging: boolean) => void;
}

function ShadersPanel({ adjustments, setAdjustments, onDragStateChange }: ShadersPanelProps) {
  const { t } = useTranslation();
  const [shaders, setShaders] = useState<Array<UserShaderInfo>>([]);
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  const seenRevisions = useRef<Record<string, string> | null>(null);
  const passes = useMemo(() => adjustments.shaderPasses ?? [], [adjustments.shaderPasses]);
  const passesRef = useRef(passes);
  passesRef.current = passes;

  // Re-scan the folder. If a file used by this image changed on disk, re-render the
  // preview: the adjustments themselves didn't change, so nothing else would.
  const refresh = useCallback(async (forceRender = false) => {
    try {
      const list = await invoke<Array<UserShaderInfo>>(Invokes.ListUserShaders);
      setShaders(list);
      const revisions = Object.fromEntries(list.map((s) => [s.file, s.revision]));
      const previous = seenRevisions.current;
      seenRevisions.current = revisions;
      const usedChanged =
        previous !== null && passesRef.current.some((pass) => previous[pass.file] !== revisions[pass.file]);
      if (forceRender || usedChanged) {
        const { setEditor, displayRevision } = useEditorStore.getState();
        setEditor({ displayRevision: displayRevision + 1 });
      }
    } catch (err) {
      console.error('Failed to list user shaders:', err);
    }
  }, []);

  useEffect(() => {
    refresh();
    const onFocus = () => refresh();
    window.addEventListener('focus', onFocus);
    return () => window.removeEventListener('focus', onFocus);
  }, [refresh]);

  const shadersByFile = useMemo(() => new Map(shaders.map((s) => [s.file, s])), [shaders]);

  const updatePasses = (update: (passes: Array<ShaderPass>) => Array<ShaderPass>) =>
    setAdjustments((prev) => ({ ...prev, shaderPasses: update(prev.shaderPasses ?? []) }));

  const addPass = (file: string) => {
    const shader = shadersByFile.get(file);
    if (!shader) return;
    updatePasses((list) => [...list, { id: uuidv4(), file, enabled: true, params: defaultParams(shader) }]);
  };

  const movePass = (index: number, delta: number) =>
    updatePasses((list) => {
      const target = index + delta;
      if (target < 0 || target >= list.length) return list;
      const next = [...list];
      [next[index], next[target]] = [next[target], next[index]];
      return next;
    });

  // "Whole image" first, then every mask by name; hidden ones still listed so a choice
  // survives toggling the mask off (the shader then sees no coverage).
  const maskOptions = useMemo(
    () => [
      { label: t('adjustments.shaders.wholeImage'), value: '' },
      ...(adjustments.masks ?? []).map((mask, index) => {
        const name = mask.name || `${t('adjustments.shaders.mask')} ${index + 1}`;
        const hidden = !mask.visible || mask.subMasks.length === 0;
        return { label: hidden ? t('adjustments.shaders.hiddenMask', { name }) : name, value: mask.id };
      }),
    ],
    [adjustments.masks, t],
  );

  const setMaskParam = (passId: string, spec: ShaderParamSpec, maskId: string) =>
    updatePasses((list) =>
      list.map((pass) => (pass.id === passId ? { ...pass, params: { ...pass.params, [spec.name]: maskId } } : pass)),
    );

  const setParam = (passId: string, spec: ShaderParamSpec, component: number, value: number) =>
    updatePasses((list) =>
      list.map((pass) => {
        if (pass.id !== passId) return pass;
        const nextValue =
          spec.components === 1
            ? value
            : Array.from({ length: spec.components }, (_, i) => (i === component ? value : paramValue(pass, spec, i)));
        return { ...pass, params: { ...pass.params, [spec.name]: nextValue } };
      }),
    );

  const iconButton = 'p-1 rounded-full text-text-secondary hover:bg-bg-primary disabled:opacity-30';

  return (
    <div className="flex flex-col gap-2">
      <Text variant={TextVariants.small}>
        {passes.length > 0 ? t('adjustments.shaders.hint') : t('adjustments.shaders.empty')}
      </Text>

      {passes.map((pass, index) => {
        const shader = shadersByFile.get(pass.file);
        const error = shader ? shader.error : t('adjustments.shaders.missingFile', { file: pass.file });
        const isCollapsed = collapsed[pass.id] ?? false;
        return (
          <div key={pass.id} className="p-1 bg-bg-tertiary rounded-md">
            <div className="flex items-center gap-1 select-none">
              <VisibilityToggle
                isVisible={pass.enabled}
                onToggle={() =>
                  updatePasses((list) => list.map((p) => (p.id === pass.id ? { ...p, enabled: !p.enabled } : p)))
                }
              />
              <button
                className="grow min-w-0 text-left truncate"
                data-tooltip={shader?.description || pass.file}
                onClick={() => setCollapsed((state) => ({ ...state, [pass.id]: !isCollapsed }))}
              >
                <Text
                  as="span"
                  variant={TextVariants.heading}
                  className={clsx('truncate', !pass.enabled && 'opacity-50')}
                >
                  {shader?.name ?? pass.file}
                </Text>
              </button>
              <button
                className={iconButton}
                disabled={index === 0}
                onClick={() => movePass(index, -1)}
                data-tooltip={t('adjustments.shaders.moveUp')}
              >
                <ChevronUp size={16} />
              </button>
              <button
                className={iconButton}
                disabled={index === passes.length - 1}
                onClick={() => movePass(index, 1)}
                data-tooltip={t('adjustments.shaders.moveDown')}
              >
                <ChevronDown size={16} />
              </button>
              <button
                className={iconButton}
                onClick={() => updatePasses((list) => list.filter((p) => p.id !== pass.id))}
                data-tooltip={t('adjustments.shaders.remove')}
              >
                <X size={16} />
              </button>
            </div>

            {error && (
              <div className="flex gap-1.5 px-1 pt-1 items-start">
                <AlertTriangle size={14} className="text-red-400 shrink-0 mt-0.5" />
                <Text variant={TextVariants.small} color={TextColors.error} className="whitespace-pre-wrap break-words">
                  {error}
                </Text>
              </div>
            )}

            {!isCollapsed && shader && shader.params.length > 0 && (
              <div className={clsx('pt-2 transition-opacity duration-300', !pass.enabled && HIDDEN_CONTENT_CLASS)}>
                {shader.params.flatMap((spec) =>
                  spec.mask
                    ? [
                        <div key={spec.name} className="flex items-center gap-2 px-1 py-1">
                          <Text variant={TextVariants.label} className="shrink-0">
                            {spec.label}
                          </Text>
                          <Dropdown<string>
                            className="grow min-w-0"
                            value={maskValue(pass, spec)}
                            placeholder={t('adjustments.shaders.missingMask')}
                            options={maskOptions}
                            onChange={(maskId) => setMaskParam(pass.id, spec, maskId)}
                          />
                        </div>,
                      ]
                    : spec.options.length > 0
                      ? [
                          <div key={spec.name} className="flex items-center gap-2 px-1 py-1">
                            <Text variant={TextVariants.label} className="shrink-0">
                              {spec.label}
                            </Text>
                            <Dropdown<number>
                              className="grow min-w-0"
                              value={paramValue(pass, spec, 0)}
                              options={spec.options.map((label, value) => ({ label, value }))}
                              onChange={(value) => setParam(pass.id, spec, 0, value)}
                            />
                          </div>,
                        ]
                    : spec.toggle
                      ? [
                          <Switch
                            key={spec.name}
                            className="px-1 py-1"
                            label={spec.label}
                            checked={paramValue(pass, spec, 0) !== 0}
                            onChange={(checked) => setParam(pass.id, spec, 0, checked ? 1 : 0)}
                          />,
                        ]
                      : Array.from({ length: spec.components }, (_, component) => (
                          <Slider
                            key={`${spec.name}.${component}`}
                            label={spec.components === 1 ? spec.label : `${spec.label} ${COMPONENT_NAMES[component]}`}
                            min={spec.min}
                            max={spec.max}
                            step={spec.step}
                            value={paramValue(pass, spec, component)}
                            defaultValue={spec.default[component]}
                            fillOrigin="min"
                            onChange={(e) => setParam(pass.id, spec, component, Number(e.target.value))}
                            onDragStateChange={onDragStateChange}
                          />
                        )),
                )}
              </div>
            )}
          </div>
        );
      })}

      <div className="flex items-center gap-1">
        <div className="grow min-w-0" onPointerDownCapture={() => refresh()}>
          <Dropdown<string>
            value={null}
            placeholder={t('adjustments.shaders.addPass')}
            options={
              shaders.length > 0
                ? shaders.map((s) => ({ label: s.error ? `${s.name} ⚠` : s.name, value: s.file }))
                : [{ label: t('adjustments.shaders.noShaders'), value: '', disabled: true }]
            }
            onChange={addPass}
          />
        </div>
        <button
          className={iconButton}
          onClick={() => invoke(Invokes.OpenUserShadersFolder).catch(console.error)}
          data-tooltip={t('adjustments.shaders.openFolder')}
        >
          <FolderOpen size={16} />
        </button>
        <button className={iconButton} onClick={() => refresh(true)} data-tooltip={t('adjustments.shaders.reload')}>
          <RefreshCw size={16} />
        </button>
      </div>
    </div>
  );
}

export default memo(ShadersPanel, areSectionPropsEqual('shaders'));
