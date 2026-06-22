import React from 'react';
import { atmosPanel, atmosHeading, glowingText } from '../../lib/atmos-styles';
import clsx from 'clsx';

interface AtmosPanelProps {
  children: React.ReactNode;
  title?: string;
  subtitle?: string;
  className?: string;
  glowColor?: 'blue' | 'cyan' | 'purple';
}

/**
 * ATMOS-styled panel wrapper
 * Provides consistent dark theme with glowing effects
 */
export const AtmosPanel: React.FC<AtmosPanelProps> = ({
  children,
  title,
  subtitle,
  className,
  glowColor = 'blue',
}) => {
  return (
    <div className={clsx(atmosPanel, className)}>
      {(title || subtitle) && (
        <div className="mb-4">
          {title && (
            <h3 className={clsx(
              'font-orbitron text-lg font-bold uppercase tracking-wider',
              'text-[#f0f4ff]',
              glowingText(glowColor)
            )}>
              {title}
            </h3>
          )}
          {subtitle && (
            <p className="text-sm text-[#a0a8d8] mt-1">{subtitle}</p>
          )}
        </div>
      )}
      {children}
    </div>
  );
};

interface AtmosButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  children: React.ReactNode;
  variant?: 'primary' | 'secondary' | 'accent';
  size?: 'sm' | 'md' | 'lg';
}

/**
 * ATMOS-styled button
 */
export const AtmosButton: React.FC<AtmosButtonProps> = ({
  children,
  variant = 'primary',
  size = 'md',
  className,
  ...props
}) => {
  const sizeClasses = {
    sm: 'px-2 py-1 text-sm',
    md: 'px-4 py-2 text-base',
    lg: 'px-6 py-3 text-lg',
  };

  const variantClasses = {
    primary: 'bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600',
    secondary: 'bg-[#151c3a] border border-[#1e2555] text-[#f0f4ff] hover:border-blue-500',
    accent: 'bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600',
  };

  return (
    <button
      className={clsx(
        'font-semibold rounded-md transition-all duration-300',
        'hover:shadow-[0_0_15px_rgba(96,165,250,0.5)]',
        'active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed',
        sizeClasses[size],
        variantClasses[variant],
        'text-white',
        className
      )}
      {...props}
    >
      {children}
    </button>
  );
};

interface AtmosBadgeProps {
  children: React.ReactNode;
  color?: 'blue' | 'cyan' | 'purple' | 'pink';
  className?: string;
}

/**
 * ATMOS-styled badge
 */
export const AtmosBadge: React.FC<AtmosBadgeProps> = ({
  children,
  color = 'blue',
  className,
}) => {
  const colorClasses = {
    blue: 'bg-blue-900/30 border-blue-500/50 text-blue-300',
    cyan: 'bg-cyan-900/30 border-cyan-500/50 text-cyan-300',
    purple: 'bg-purple-900/30 border-purple-500/50 text-purple-300',
    pink: 'bg-pink-900/30 border-pink-500/50 text-pink-300',
  };

  return (
    <span
      className={clsx(
        'inline-block px-3 py-1 rounded-full text-sm font-mono',
        'border',
        'hover:shadow-[0_0_10px_rgba(96,165,250,0.4)]',
        'transition-all duration-300',
        colorClasses[color],
        className
      )}
    >
      {children}
    </span>
  );
};

interface AtmosHeadingProps {
  level: 1 | 2 | 3 | 4 | 5 | 6;
  children: React.ReactNode;
  glow?: boolean;
  glowColor?: 'blue' | 'cyan' | 'purple';
  className?: string;
}

/**
 * ATMOS-styled heading
 */
export const AtmosHeading: React.FC<AtmosHeadingProps> = ({
  level,
  children,
  glow = true,
  glowColor = 'blue',
  className,
}) => {
  const sizeClasses = {
    1: 'text-4xl',
    2: 'text-3xl',
    3: 'text-2xl',
    4: 'text-xl',
    5: 'text-lg',
    6: 'text-base',
  };

  const Component = `h${level}` as const;

  return React.createElement(
    Component,
    {
      className: clsx(
        'font-orbitron font-bold uppercase tracking-wider',
        'text-[#f0f4ff]',
        sizeClasses[level],
        glow && glowingText(glowColor),
        className
      ),
    },
    children
  );
};

interface AtmosMetricProps {
  label: string;
  value: string | number;
  unit?: string;
  trend?: 'up' | 'down' | 'neutral';
  className?: string;
}

/**
 * ATMOS-styled metric display
 */
export const AtmosMetric: React.FC<AtmosMetricProps> = ({
  label,
  value,
  unit,
  trend,
  className,
}) => {
  const trendColor = {
    up: 'text-green-400',
    down: 'text-red-400',
    neutral: 'text-cyan-400',
  };

  return (
    <div className={clsx('flex flex-col gap-1', className)}>
      <span className="text-xs uppercase tracking-wider text-[#a0a8d8]">
        {label}
      </span>
      <div className={clsx(
        'font-mono text-2xl font-bold',
        'drop-shadow-[0_0_10px_rgba(34,211,238,0.3)]',
        trend && trendColor[trend]
      )}>
        {value}
        {unit && <span className="text-lg text-[#a0a8d8]">{unit}</span>}
      </div>
    </div>
  );
};

/**
 * ATMOS-styled divider
 */
export const AtmosDivider: React.FC<{ className?: string }> = ({ className }) => (
  <div className={clsx(
    'h-px bg-gradient-to-r from-transparent via-[#1e2555] to-transparent',
    'my-4',
    className
  )} />
);

/**
 * ATMOS-styled toast/notification
 */
export const AtmosNotification: React.FC<{
  type: 'info' | 'success' | 'warning' | 'error';
  message: string;
  onClose?: () => void;
}> = ({ type, message, onClose }) => {
  const typeStyles = {
    info: 'bg-blue-900/30 border-blue-500/50 text-blue-300',
    success: 'bg-green-900/30 border-green-500/50 text-green-300',
    warning: 'bg-yellow-900/30 border-yellow-500/50 text-yellow-300',
    error: 'bg-red-900/30 border-red-500/50 text-red-300',
  };

  return (
    <div className={clsx(
      'p-4 rounded-lg border',
      'flex justify-between items-center gap-3',
      'animate-in fade-in slide-in-from-top-2',
      typeStyles[type]
    )}>
      <span className="font-rajdhani font-medium">{message}</span>
      {onClose && (
        <button
          onClick={onClose}
          className="text-lg hover:opacity-70 transition-opacity"
        >
          ×
        </button>
      )}
    </div>
  );
};
