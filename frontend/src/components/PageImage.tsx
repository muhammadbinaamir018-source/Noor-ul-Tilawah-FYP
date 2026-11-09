import { useEffect, useRef, useState, type ReactNode } from 'react';

export interface RenderedSize {
  width: number;
  height: number;
}

interface Props {
  src: string;
  children?: (size: RenderedSize) => ReactNode;
  overlayProps?: React.HTMLAttributes<HTMLDivElement>;
  overlayRef?: React.Ref<HTMLDivElement>;
}

/**
 * Renders the scanned page image and an overlay layer that exactly covers it,
 * re-measuring on resize so fractional region rects stay aligned at any zoom.
 */
export function PageImage({ src, children, overlayProps, overlayRef }: Props) {
  const imgRef = useRef<HTMLImageElement>(null);
  const [size, setSize] = useState<RenderedSize>({ width: 0, height: 0 });

  useEffect(() => {
    const img = imgRef.current;
    if (!img) return;
    const measure = () =>
      setSize({ width: img.clientWidth, height: img.clientHeight });
    const observer = new ResizeObserver(measure);
    observer.observe(img);
    if (img.complete) measure();
    return () => observer.disconnect();
  }, [src]);

  return (
    <div className="page-image-wrap">
      <img
        ref={imgRef}
        src={src}
        alt="Qaida page"
        draggable={false}
        onLoad={(e) =>
          setSize({ width: e.currentTarget.clientWidth, height: e.currentTarget.clientHeight })
        }
      />
      <div className="page-overlay" ref={overlayRef} {...overlayProps}>
        {size.width > 0 && children?.(size)}
      </div>
    </div>
  );
}
