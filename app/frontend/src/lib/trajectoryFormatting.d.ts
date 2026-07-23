export const HKT_TIME_ZONE: string

export function timestampToEpochMillis(timestamp: string | number | null | undefined): number | null
export function formatHktTime(timestamp: string | number | null | undefined): string
export function formatHktDateTime(timestamp: string | number | null | undefined): string
export function formatAge(seconds: number | null | undefined): string
export function buildTrajectoryTooltip(args: {
  timestamp: string | number | null | undefined
  metadata?: object
  entries: Array<{ seriesName: string; color: string; value: number | null }>
  expectedCycleIntervalSeconds?: number | null
}): string
export function canConnectRealModelCycles(
  previousTimestamp: string | null | undefined,
  nextTimestamp: string | null | undefined,
  expectedCycleIntervalSeconds: number | null | undefined,
): boolean
