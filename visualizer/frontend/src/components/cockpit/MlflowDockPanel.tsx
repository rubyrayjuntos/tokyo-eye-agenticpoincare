/** Embed MLflow UI inside Dockview. */
export default function MlflowDockPanel() {
  return (
    <iframe
      title="MLflow"
      src="/mlflow/"
      className="h-full w-full border-0 bg-zinc-950"
      sandbox="allow-same-origin allow-scripts allow-forms allow-popups"
    />
  );
}
