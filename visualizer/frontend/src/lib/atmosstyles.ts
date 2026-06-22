/**
 * ATMOS-style component utilities
 * Provides tailwind class combinations for ATMOS aesthetic
 */

export const atmosColors = {
  bg: {
    primary: 'bg-[#0a0e27]',
    secondary: 'bg-[#0f1535]',
    tertiary: 'bg-[#151c3a]',
  },
  text: {
    primary: 'text-[#f0f4ff]',
    secondary: 'text-[#a0a8d8]',
  },
  accent: {
    blue: 'text-blue-400',
    cyan: 'text-cyan-400',
    purple: 'text-purple-400',
    pink: 'text-pink-400',
  },
  border: 'border-[#1e2555]',
};

/**
 * Panel with glowing border effect
 */
export const atmosPanel = `
  ${atmosColors.bg.secondary}
  border ${atmosColors.border}
  rounded-lg
  p-4
  shadow-lg
  transition-all duration-300
  hover:shadow-[0_0_20px_rgba(34,211,238,0.3)]
`;

/**
 * Button with ATMOS styling
 */
export const atmosButton = `
  px-4 py-2
  font-semibold
  rounded-md
  bg-gradient-to-r from-blue-500 to-cyan-500
  hover:from-blue-600 hover:to-cyan-600
  text-white
  transition-all duration-300
  hover:shadow-[0_0_15px_rgba(96,165,250,0.5)]
  active:scale-95
  disabled:opacity-50 disabled:cursor-not-allowed
`;

/**
 * Badge with glow effect
 */
export const atmosBadge = `
  inline-block
  px-3 py-1
  rounded-full
  text-sm font-mono
  bg-blue-900/30
  border border-blue-500/50
  text-blue-300
  hover:shadow-[0_0_10px_rgba(96,165,250,0.4)]
  transition-all duration-300
`;

/**
 * Input with glow focus state
 */
export const atmosInput = `
  w-full
  px-3 py-2
  rounded-md
  bg-[#0f1535]
  border border-[#1e2555]
  text-[#f0f4ff]
  placeholder-[#a0a8d8]/50
  focus:border-cyan-400
  focus:shadow-[0_0_10px_rgba(34,211,238,0.4)]
  transition-all duration-300
`;

/**
 * Heading with ATMOS styling
 */
export const atmosHeading = {
  h1: 'font-orbitron text-3xl font-bold text-[#f0f4ff] uppercase tracking-wider',
  h2: 'font-orbitron text-2xl font-bold text-[#f0f4ff] uppercase tracking-wider',
  h3: 'font-rajdhani text-xl font-bold text-[#f0f4ff] uppercase tracking-wider',
};

/**
 * Metric display with monospace font and glow
 */
export const atmosMetric = `
  font-mono
  text-sm
  font-bold
  text-cyan-300
  drop-shadow-[0_0_10px_rgba(34,211,238,0.3)]
`;

/**
 * Grid layout for dashboard
 */
export const atmosGrid = `
  grid gap-4
  auto-rows-max
  overflow-auto
`;

/**
 * Icon button with hover effect
 */
export const atmosIconButton = `
  p-2
  rounded-md
  text-[#a0a8d8]
  hover:text-cyan-400
  hover:bg-[#151c3a]
  transition-all duration-300
  cursor-pointer
`;

/**
 * Apply ATMOS font family
 */
export const atmosFont = 'font-rajdhani';
export const atmosFontHeading = 'font-orbitron';
export const atmosFontMono = 'font-mono';

/**
 * Generate a glowing text effect class
 */
export const glowingText = (color: 'blue' | 'cyan' | 'purple' = 'blue') => {
  const glows = {
    blue: 'drop-shadow-[0_0_10px_rgba(96,165,250,0.5)]',
    cyan: 'drop-shadow-[0_0_10px_rgba(34,211,238,0.5)]',
    purple: 'drop-shadow-[0_0_10px_rgba(167,139,250,0.5)]',
  };
  return glows[color];
};

/**
 * Combine ATMOS panel + button styles
 */
export const atmosCard = `
  ${atmosPanel}
  backdrop-blur-md
  bg-opacity-80
`;

/**
 * ATMOS-style table row
 */
export const atmosTableRow = `
  border-b border-[#1e2555]
  hover:bg-[#151c3a]
  transition-colors duration-200
`;

/**
 * Animated border effect
 */
export const atmosBorderAnimation = `
  relative
  after:content-['']
  after:absolute
  after:inset-0
  after:border after:border-cyan-400
  after:rounded-lg
  after:opacity-0
  after:animate-pulse
`;
