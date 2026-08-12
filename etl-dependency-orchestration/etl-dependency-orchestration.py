def schedule_pipeline(tasks, resource_budget):
    """
    Schedule ETL tasks respecting dependencies and resource limits.
    """
    # Write code here
    task_map = {task["name"]: task for task in tasks}

    # Validate that no task individually exceeds the budget
    for task in tasks:
        if task["resources"] > resource_budget:
            raise ValueError(
                f"Task '{task['name']}' requires more resources "
                f"than the available budget."
            )

    completed = set()
    running = {}  # task_name -> end_time
    start_times = []

    time = 0
    used_resources = 0

    while len(completed) < len(tasks):

        # --------------------------------------------------
        # 1. Complete all tasks that have finished by now
        # --------------------------------------------------
        finished = [
            name
            for name, end_time in running.items()
            if end_time <= time
        ]

        for name in finished:
            task = task_map[name]

            used_resources -= task["resources"]
            completed.add(name)
            del running[name]

        # --------------------------------------------------
        # 2. Find tasks whose dependencies are completed
        # --------------------------------------------------
        ready = []

        for task in tasks:
            name = task["name"]

            # Already completed or running
            if name in completed or name in running:
                continue

            dependencies = task.get("depends_on", [])

            if all(dep in completed for dep in dependencies):
                ready.append(task)

        # Alphabetical ordering
        ready.sort(key=lambda task: task["name"])

        # --------------------------------------------------
        # 3. Schedule as many ready tasks as resources allow
        # --------------------------------------------------
        scheduled_any = False

        for task in ready:
            required = task["resources"]

            # Skip if this task does not fit.
            # Continue checking later ready tasks.
            if used_resources + required > resource_budget:
                continue

            name = task["name"]

            start_time = time
            end_time = time + task["duration"]

            start_times.append((name, start_time))

            running[name] = end_time
            used_resources += required

            scheduled_any = True

        # --------------------------------------------------
        # 4. If something is running, jump to its next completion
        # --------------------------------------------------
        if running:
            next_completion = min(running.values())

            # If we scheduled something, it may have the same
            # current time, so move directly to its completion.
            time = next_completion

        # --------------------------------------------------
        # 5. Nothing was scheduled and nothing is running
        #    => circular/invalid dependency
        # --------------------------------------------------
        elif not scheduled_any and len(completed) < len(tasks):
            raise ValueError("Circular or invalid dependency detected.")

    return start_times