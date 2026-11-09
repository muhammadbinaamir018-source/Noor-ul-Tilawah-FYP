import type { Rect } from '../content/types';

interface Props {
  rect: Rect;
  className?: string;
  title?: string;
  onMouseEnter?: () => void;
  onMouseLeave?: () => void;
  onClick?: () => void;
  children?: React.ReactNode;
}

export function RegionBox({ rect, className = '', title, children, ...handlers }: Props) {
  return (
    <div
      className={`region-box ${className}`}
      title={title}
      style={{
        left: `${rect.x * 100}%`,
        top: `${rect.y * 100}%`,
        width: `${rect.w * 100}%`,
        height: `${rect.h * 100}%`,
      }}
      {...handlers}
    >
      {children}
    </div>
  );
}
