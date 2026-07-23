export const HKT_TIME_ZONE = "Asia/Hong_Kong"

const DATE_TIME_FORMATTER = new Intl.DateTimeFormat("en-GB", {
  timeZone: HKT_TIME_ZONE,
  day: "2-digit",
  month: "short",
  year: "numeric",
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
})

const TIME_FORMATTER = new Intl.DateTimeFormat("en-GB", {
  timeZone: HKT_TIME_ZONE,
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
})

function explicitlyZonedTimestamp(timestamp) {
  const value = String(timestamp || "").trim()
  if (!value) return ""
  if (/[zZ]$|[+-]\d{2}:?\d{2}$/.test(value)) return value
  if (/^\d{4}-\d{2}-\d{2}$/.test(value)) return `${value}T00:00:00+08:00`
  return `${value}+08:00`
}

export function timestampToEpochMillis(timestamp) {
  if (typeof timestamp === "number") return Number.isFinite(timestamp) ? timestamp : null
  const value = new Date(explicitlyZonedTimestamp(timestamp)).getTime()
  return Number.isFinite(value) ? value : null
}

export function formatHktTime(timestamp) {
  const epoch = timestampToEpochMillis(timestamp)
  return epoch == null ? "—" : TIME_FORMATTER.format(epoch)
}

export function formatHktDateTime(timestamp) {
  const epoch = timestampToEpochMillis(timestamp)
  return epoch == null ? "—" : `${DATE_TIME_FORMATTER.format(epoch)} HKT`
}

export function formatAge(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "—"
  if (seconds < 60) return `${Math.round(seconds)}s`
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`
  return `${(seconds / 3600).toFixed(seconds % 3600 === 0 ? 0 : 1)}h`
}

function escapeHtml(value) {
  return String(value ?? "—")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;")
}

function metadataRow(label, value) {
  return `<div style="display:flex;justify-content:space-between;gap:20px;margin-top:3px"><span style="color:rgba(255,255,255,.52)">${escapeHtml(label)}</span><span style="color:#fff">${escapeHtml(value)}</span></div>`
}

export function buildTrajectoryTooltip({ timestamp, metadata, entries, expectedCycleIntervalSeconds }) {
  const point = metadata || {}
  const actualObservation = point.actual_observation_timestamp || point.weather_data_through
  const actualFirstSeen = point.actual_first_seen_timestamp || point.weather_first_seen_timestamp
  const values = entries
    .filter((entry) => entry.value != null && Number.isFinite(entry.value))
    .map((entry) => `<div style="display:flex;justify-content:space-between;gap:20px;margin-top:3px"><span style="color:${escapeHtml(entry.color)}">${escapeHtml(entry.seriesName)}</span><span style="color:#fff;font-weight:600">${entry.value.toFixed(2)}°C</span></div>`)
    .join("")
  const threshold = Number.isFinite(expectedCycleIntervalSeconds) && expectedCycleIntervalSeconds > 0
    ? expectedCycleIntervalSeconds * 2
    : null
  const modelStale = threshold != null && Number.isFinite(point.model_age_seconds)
    ? point.model_age_seconds > threshold
    : null
  const lineage = [
    metadataRow("Actual observation", formatHktDateTime(actualObservation)),
    metadataRow("Actual first seen", formatHktDateTime(actualFirstSeen)),
    metadataRow("Actual source release", formatHktDateTime(point.actual_source_release_timestamp)),
    metadataRow("Release lag", formatAge(point.actual_release_lag_seconds)),
    metadataRow("Prediction decision", formatHktDateTime(point.prediction_decision_timestamp)),
    metadataRow("Weather data through", formatHktDateTime(point.weather_data_through)),
    metadataRow("Weather first seen", formatHktDateTime(point.weather_first_seen_timestamp)),
    metadataRow("Weather age", formatAge(point.weather_age_seconds)),
    metadataRow("Model-cycle ID", point.model_cycle_id || "—"),
    metadataRow("Model age", formatAge(point.model_age_seconds)),
    ...(modelStale == null ? [] : [metadataRow("Model status", modelStale ? "stale" : "current")]),
  ].join("")
  return `<div style="font-family:ui-monospace,monospace;font-size:10px;color:rgba(255,255,255,.55);margin-bottom:6px">${escapeHtml(formatHktDateTime(timestamp))}</div>${values}<div style="border-top:1px solid rgba(255,255,255,.12);margin-top:7px;padding-top:4px">${lineage}</div>`
}

export function canConnectRealModelCycles(previousTimestamp, nextTimestamp, expectedCycleIntervalSeconds) {
  if (!Number.isFinite(expectedCycleIntervalSeconds) || expectedCycleIntervalSeconds <= 0) return false
  const previous = timestampToEpochMillis(previousTimestamp)
  const next = timestampToEpochMillis(nextTimestamp)
  if (previous == null || next == null || next <= previous) return false
  const previousDate = formatHktDateTime(previousTimestamp).slice(0, 11)
  const nextDate = formatHktDateTime(nextTimestamp).slice(0, 11)
  return previousDate === nextDate && (next - previous) / 1000 <= expectedCycleIntervalSeconds * 2
}

export function segmentCoordinatesByCadence(coordinates, expectedCycleIntervalSeconds) {
  const ordered = [...(coordinates || [])]
    .filter((point) => Array.isArray(point) && point.length >= 2 && Number.isFinite(point[0]) && Number.isFinite(point[1]))
    .sort((left, right) => left[0] - right[0])
  const segments = []
  let segment = []
  for (const point of ordered) {
    const previous = segment.at(-1)
    if (!previous || canConnectRealModelCycles(previous[0], point[0], expectedCycleIntervalSeconds)) {
      segment.push(point)
      continue
    }
    segments.push(segment)
    segment = [point]
  }
  if (segment.length) segments.push(segment)
  return segments
}
