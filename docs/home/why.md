# Why Floki

Floki is an open-source framework for building and orchestrating LLM-based autonomous agents, designed to simplify the complexity of creating scalable agentic workflows and microservices. Inspired by the growing need for frameworks that integrate seamlessly with distributed systems, Floki enables developers to focus on designing intelligent agents without getting bogged down by infrastructure concerns.

## The Problem

Many agentic frameworks today attempt to redefine how microservices are built and orchestrated by developing their own platforms for workflows, Pub/Sub messaging, state management, and service communication. While these efforts showcase innovation, they often lead to a steep learning curve, fragmented systems, and unnecessary complexity when scaling or adapting to new environments.

Rather than building on existing solutions that are proven to handle these challenges at scale, many frameworks require developers to adopt entirely new paradigms or recreate foundational infrastructure. This added complexity often diverts focus from the primary goal: designing and implementing intelligent, effective agents.

## Floki's Approach

Floki takes a distinct approach by building on [Dapr](https://dapr.io/), a portable and event-driven runtime optimized for distributed systems. Dapr offers built-in APIs and patterns such as state management, Pub/Sub messaging, service invocation, and virtual actors—that eliminate the need to recreate foundational components from scratch. By integrating seamlessly with Dapr, Floki empowers developers to focus on the intelligence and behavior of LLM-powered agents while leveraging a proven framework for scalability and reliability.

Rather than reinventing microservices, Floki enables developers to design, test, and deploy agents that seamlessly integrate as collaborative services within larger systems. Whether experimenting with a single agent or orchestrating workflows involving multiple agents, Floki simplifies the exploration and implementation of scalable agentic workflows.

## Conclusion

Floki provides a unified framework for designing, deploying, and orchestrating LLM-powered agents. By leveraging Dapr’s runtime and modular components, Floki allows developers to focus on building intelligent systems without worrying about the complexities of distributed infrastructure. Whether you're creating standalone agents or orchestrating multi-agent workflows, Floki empowers you to explore the future of intelligent, scalable, and collaborative systems.

## Why the Name Floki?

The name `Floki` is inspired by both history and fiction. Historically, [Floki Vilgerðarson](https://en.wikipedia.org/wiki/Hrafna-Fl%C3%B3ki_Vilger%C3%B0arson) is known in Norse sagas as the first Norseman to journey to Iceland, embodying a spirit of discovery. In the [Vikings series](https://en.wikipedia.org/wiki/Vikings_(2013_TV_series)), Floki is portrayed as a skilled boat builder, creating vessels that allowed his people to explore and achieve their goals.

In the same way, this framework equips developers with the tools to build, prototype, and deploy their own agents or fleets of agents, enabling them to experiment and explore the potential of LLM-based workflows.