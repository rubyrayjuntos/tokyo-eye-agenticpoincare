import type {
  EventRegistry,
  MiddlewareNext,
  OrchestrationSnapshotPayload,
  SubscriptionCallback,
  WorkbenchMiddleware,
  WorkbenchTopic,
} from "./eventRegistry";

export class WorkbenchBus {
  private subscribers = new Map<string, Set<SubscriptionCallback<any>>>();
  private wildcardSubscribers = new Set<(topic: string, payload: unknown) => void>();
  private stateCache = new Map<string, unknown>();
  private middlewares: WorkbenchMiddleware[] = [];

  constructor() {
    this.use((topic, payload, next) => {
      if (!String(topic).startsWith("tool:")) {
        next();
        return;
      }

      const snapshot = this.getLatestState("system:orchestration_snapshot");
      const toolName =
        String(topic) === "tool:call"
          ? (payload as EventRegistry["tool:call"]).tool
          : String(topic).replace("tool:call:", "");

      const blocked = snapshot?.policy?.blocked_tools ?? [];
      const allowed = snapshot?.policy?.allowed_tools ?? [];

      if (blocked.includes(toolName)) {
        this.publish("system:warning", {
          message: `Tool '${toolName}' is blocked in phase '${snapshot?.discovery_phase ?? "unknown"}'.`,
          timestamp: new Date().toISOString(),
        });
        return;
      }

      if (allowed.length > 0 && !allowed.includes(toolName)) {
        this.publish("system:warning", {
          message: `Tool '${toolName}' is not permitted in the current discovery phase.`,
          timestamp: new Date().toISOString(),
        });
        return;
      }

      next();
    });
  }

  use(middleware: WorkbenchMiddleware): void {
    this.middlewares.push(middleware);
  }

  subscribe<T extends WorkbenchTopic>(
    topic: T,
    callback: SubscriptionCallback<T>,
  ): () => void {
    const key = String(topic);
    if (!this.subscribers.has(key)) {
      this.subscribers.set(key, new Set());
    }
    this.subscribers.get(key)!.add(callback as SubscriptionCallback<any>);
    return () => {
      const set = this.subscribers.get(key);
      if (!set) return;
      set.delete(callback as SubscriptionCallback<any>);
      if (set.size === 0) this.subscribers.delete(key);
    };
  }

  subscribeToAll(callback: (topic: string, payload: unknown) => void): () => void {
    this.wildcardSubscribers.add(callback);
    return () => this.wildcardSubscribers.delete(callback);
  }

  getLatestState<T extends WorkbenchTopic>(topic: T): EventRegistry[T] | undefined {
    return this.stateCache.get(String(topic)) as EventRegistry[T] | undefined;
  }

  setOrchestrationSnapshot(snapshot: OrchestrationSnapshotPayload): void {
    this.stateCache.set("system:orchestration_snapshot", snapshot);
  }

  publish<T extends WorkbenchTopic>(
    topic: T,
    payload: EventRegistry[T],
  ): { accepted: boolean } {
    let index = 0;
    let accepted = false;

    const runMiddleware = () => {
      if (index < this.middlewares.length) {
        const middleware = this.middlewares[index++];
        middleware(topic, payload, runMiddleware);
      } else {
        accepted = true;
        this.executeDispatch(topic, payload);
      }
    };

    runMiddleware();
    return { accepted };
  }

  clearCache(): void {
    this.stateCache.clear();
  }

  private executeDispatch<T extends WorkbenchTopic>(topic: T, payload: EventRegistry[T]): void {
    this.stateCache.set(String(topic), payload);

    this.subscribers.forEach((callbacks, subTopic) => {
      if (this.isTopicMatch(subTopic, String(topic))) {
        callbacks.forEach((callback) => {
          try {
            callback(payload);
          } catch (error) {
            console.error(`WorkbenchBus error on topic [${String(topic)}]:`, error);
          }
        });
      }
    });

    this.wildcardSubscribers.forEach((callback) => {
      try {
        callback(String(topic), payload);
      } catch (error) {
        console.error("WorkbenchBus wildcard listener error:", error);
      }
    });

    if (String(topic) !== "broker:log_added") {
      this.publish("broker:log_added", {
        id: Math.random().toString(36).slice(2, 9),
        topic: String(topic),
        payload,
        timestamp: new Date().toISOString(),
        type: String(topic).startsWith("tool:")
          ? "tool_call"
          : String(topic).startsWith("ui:")
            ? "interaction"
            : String(topic).includes("warning")
              ? "warning"
              : "system",
      });
    }
  }

  private isTopicMatch(subscriberTopic: string, publishedTopic: string): boolean {
    if (subscriberTopic === publishedTopic) return true;
    if (subscriberTopic === "*") return true;
    if (subscriberTopic.endsWith("*")) {
      return publishedTopic.startsWith(subscriberTopic.slice(0, -1));
    }
    return false;
  }
}
