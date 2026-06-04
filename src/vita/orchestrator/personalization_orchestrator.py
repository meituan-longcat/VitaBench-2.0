"""
PersonalizationOrchestrator: Manages the multi-subtask personalization workflow.

Workflow:
1. For each subtask:
   a. Process interactions (update memory)
   b. Set up environment for this subtask's domain
   c. Run agent-user conversation until subtask completes
   d. Evaluate subtask
2. Aggregate results
"""

import time as time_module
import uuid
from copy import deepcopy
from typing import Any, Dict, List, Optional

from loguru import logger

from vita.agent.base import BaseAgent, is_valid_agent_history_message
from vita.agent.personalization_agent import PersonalizationAgent
from vita.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolMessage,
    UserMessage,
)
from vita.data_model.personalization_task import PersonalizationTask, SubTask
from vita.data_model.simulation import (
    RewardInfo,
    SimulationRun,
    TerminationReason,
)
from vita.data_model.tasks import Task, EvaluationCriteria, UserScenario
from vita.environment.db import DB
from vita.environment.environment import Environment, get_cross_environment
from vita.evaluator.evaluator import evaluate_simulation
from vita.orchestrator.orchestrator import Orchestrator, get_default_first_agent_message, Role
from vita.user.personalization_user import PersonalizationUser
from vita.user.user_simulator import UserSimulator
from vita.utils.llm_utils import get_cost
from vita.utils.utils import get_now, format_time, global_time


class PersonalizationOrchestrator:
    """Manages the personalization workflow across multiple sequential subtasks.

    The orchestrator:
    1. For each subtask, updates memory, sets up environment, and runs conversation
    2. Evaluates each subtask independently
    3. Aggregates results across all subtasks
    """

    def __init__(
        self,
        task: PersonalizationTask,
        agent: PersonalizationAgent,
        user: PersonalizationUser,
        max_steps_per_subtask: int = 100,
        max_errors: int = 10,
        seed: Optional[int] = None,
        evaluation_type: str = "trajectory",
        llm_evaluator: Optional[str] = None,
        llm_args_evaluator: Optional[dict] = None,
        language: str = None,
        enable_outcome_reward: bool = False,
    ):
        """Initialize the PersonalizationOrchestrator.

        Args:
            task: The personalization task to execute.
            agent: The personalization agent with memory.
            user: The personalization user simulator.
            max_steps_per_subtask: Maximum steps per subtask conversation.
            max_errors: Maximum consecutive errors allowed.
            seed: Random seed.
            evaluation_type: Type of evaluation to use.
            llm_evaluator: LLM model for evaluation.
            llm_args_evaluator: LLM arguments for evaluation.
            language: Language setting.
            enable_outcome_reward: If True, combine trajectory reward with action-level
                outcome reward via min(traj, outcome) per subtask.
        """
        self.task = task
        self.agent = agent
        self.user = user
        self.max_steps_per_subtask = max_steps_per_subtask
        self.max_errors = max_errors
        self.seed = seed
        self.evaluation_type = evaluation_type
        self.llm_evaluator = llm_evaluator
        self.llm_args_evaluator = llm_args_evaluator
        self.language = language
        self.enable_outcome_reward = enable_outcome_reward

    def run(self) -> SimulationRun:
        """Run the full personalization simulation.

        Returns:
            SimulationRun with aggregated results from all subtasks.
        """
        overall_start_time = get_now()
        overall_start = time_module.perf_counter()

        print("\n" + "=" * 80)
        print("🚀 PERSONALIZATION WORKFLOW START")
        print(f"   Task ID: {self.task.id}")
        print(f"   User: {self.task.user_profile.get('user_id', 'unknown')}")
        print(f"   Total subtasks: {len(self.task.subtasks)}")
        print("=" * 80)

        all_subtask_results: List[Dict[str, Any]] = []
        all_messages: List[Message] = []
        all_states: Dict[str, Any] = {"old_states": [], "new_states": []}
        all_reward_infos: List[Optional[RewardInfo]] = []

        # Phase 2: Execute each subtask
        for i, subtask in enumerate(self.task.subtasks):
            print("\n" + "=" * 60)
            print(f"📋 PHASE 2.{i}: SUBTASK {i + 1}/{len(self.task.subtasks)}")
            print(f"   ID: {subtask.subtask_id}")
            print(f"   Domain: {subtask.domain}")
            print(f"   Instruction: {subtask.instruction}")
            print(f"   Time: {subtask.environment.get('time', 'N/A')}")
            logger.info(
                f"Phase 2.{i}: Processing subtask {i + 1}/{len(self.task.subtasks)} "
                f"(domain: {subtask.domain}, id: {subtask.subtask_id})"
            )

            # 2a-pre: If using GroundtruthMemory, inject the ground-truth preference
            from vita.memory.groundtruth_memory import GroundtruthMemory
            if isinstance(self.agent.memory, GroundtruthMemory):
                us = subtask.user_scenario
                gt_pref = getattr(us, "personalized_preference_memory", None) or {}
                historical_chat = getattr(subtask, "historical_chat", None) or []
                historical_behavior = getattr(subtask, "historical_behavior", None) or {}
                self.agent.memory.set_groundtruth(
                    gt_pref,
                    historical_chat=historical_chat,
                    historical_behavior=historical_behavior,
                )
                print(f"\n   🎯 Groundtruth preference injected for subtask {i} "
                      f"(chat={len(historical_chat)}, behavior_keys={len(historical_behavior)})")

            # Tell cache-backed memory backends which subtask file to load.
            # Other backends (null / rewrite / rag / full_context / groundtruth)
            # ignore this hook via the hasattr guard.
            if hasattr(self.agent.memory, "set_current_location"):
                self.agent.memory.set_current_location(
                    self.task.user_profile.get("user_id", self.task.id),
                    subtask.subtask_id,
                )

            # 2a: Process interactions (update memory)
            if subtask.interactions:
                print(f"\n   📥 Step 2a: Updating memory with {len(subtask.interactions)} new interactions")
                for idx, inter in enumerate(subtask.interactions[:3]):
                    if isinstance(inter, dict):
                        print(f"      [{idx+1}] date={inter.get('date', '?')}, behaviors={len(inter.get('behavior', []))}")
                    else:
                        print(f"      [{idx+1}] type={inter.type}, timestamp={inter.timestamp}")
                if len(subtask.interactions) > 3:
                    print(f"      ... and {len(subtask.interactions) - 3} more")
                logger.info(
                    f"  Updating memory with {len(subtask.interactions)} new interactions"
                )
                self.agent.process_interactions(subtask.interactions)
                memory_content = self.agent.memory.read()
                print(f"      ✅ Memory updated. Preview:")
                for line in memory_content.split("\n")[:10]:
                    print(f"         {line}")
            else:
                print(f"\n   📥 Step 2a: Skipping (no interactions)")

            # 2b: Set up environment for this subtask's domain
            print(f"\n   🔧 Step 2b: Setting up environment for domain '{subtask.domain}'")
            environment = self._setup_subtask_environment(subtask)

            # Update agent tools for the new environment
            tools = environment.get_tools()
            self.agent.update_tools(tools)
            tool_names = [t.name if hasattr(t, 'name') else str(t) for t in tools] if tools else []
            print(f"      Tools loaded: {len(tool_names)} tools")
            if tool_names:
                print(f"      Tool list: {', '.join(tool_names[:10])}")
                if len(tool_names) > 10:
                    print(f"      ... and {len(tool_names) - 10} more")

            # Update agent domain policy, but preserve personalization instructions.
            # For proactive subtasks, swap in a system prompt that does NOT bias the
            # agent toward proactively pushing purchases.
            from vita.prompts import get_prompts
            prompts = get_prompts(self.language)
            personalization_addendum = prompts.personalization_agent_addendum
            if "proactive" in subtask.skill_tested:
                base_policy_template = prompts.personalization_agent_proactive_system_prompt
            else:
                base_policy_template = prompts.personalization_agent_system_prompt
            self.agent.domain_policy = (
                base_policy_template + "\n\n" + personalization_addendum
            )

            # Update agent time for this subtask
            subtask_time = subtask.environment.get("time")
            if subtask_time:
                from vita.utils.utils import get_weekday
                self.agent.time = subtask_time + " " + get_weekday(subtask_time, self.language)
                print(f"      Agent time set to: {self.agent.time}")

            # Print the system prompt preview
            sys_prompt = self.agent.system_prompt
            print(f"\n      📄 Agent system prompt preview ({len(sys_prompt)} chars):")
            for line in sys_prompt.split("\n")[:8]:
                print(f"         {line}")
            print(f"         ...")

            # Set current instruction for memory retrieval
            self.agent.set_current_instruction(subtask.instruction)

            # Capture the memory content that will be injected into the agent's system prompt
            memory_read_content = self.agent.memory.read(query=subtask.instruction)
            print(f"\n      🧠 Memory read for this subtask ({len(memory_read_content)} chars):")
            for line in memory_read_content.split("\n")[:5]:
                print(f"         {line}")
            if memory_read_content.count("\n") > 5:
                print(f"         ...")

            # 2c: Run the subtask conversation
            print(f"\n   💬 Step 2c: Running subtask conversation...")
            subtask_result = self._run_subtask(subtask, environment, i)
            num_msgs = len(subtask_result["messages"])
            termination = subtask_result["termination_reason"]
            print(f"      ✅ Conversation completed: {num_msgs} messages, termination={termination}")

            # Print conversation summary
            print(f"\n      📜 Conversation transcript:")
            for msg in subtask_result["messages"]:
                role = msg.role if hasattr(msg, 'role') else 'unknown'
                content = getattr(msg, 'content', '')
                tool_calls = getattr(msg, 'tool_calls', None)
                if content:
                    content_preview = content[:120].replace('\n', ' ')
                    print(f"         [{role}] {content_preview}")
                if tool_calls:
                    for tc in tool_calls:
                        tc_name = tc.name if hasattr(tc, 'name') else str(tc)
                        tc_args = str(tc.arguments)[:100] if hasattr(tc, 'arguments') else ''
                        print(f"         [{role}] 🔧 {tc_name}({tc_args})")

            # 2d: Evaluate the subtask
            print(f"\n   📊 Step 2d: Evaluating subtask...")
            if subtask.evaluation_criteria is not None:
                if subtask.evaluation_criteria.overall_rubrics:
                    print(f"      Rubric items: {len(subtask.evaluation_criteria.overall_rubrics)}")
                    for r_idx, rubric in enumerate(subtask.evaluation_criteria.overall_rubrics):
                        print(f"         [{r_idx+1}] {rubric[:100]}")
                if subtask.evaluation_criteria.expected_states:
                    print(f"      Expected states: {len(subtask.evaluation_criteria.expected_states)}")
                reward_info = self._evaluate_subtask(subtask, subtask_result)
                all_reward_infos.append(reward_info)
                print(f"      ✅ Evaluation result: reward={reward_info.reward}")
                if reward_info.info:
                    print(f"      Details: {str(reward_info.info)[:200]}")
            else:
                all_reward_infos.append(None)
                print(f"      ⏭️  No evaluation criteria, skipping")

            subtask_result["memory_content"] = memory_read_content
            all_subtask_results.append(subtask_result)
            all_messages.extend(subtask_result["messages"])
            # Merge states
            if "states" in subtask_result:
                states = subtask_result["states"]
                all_states["old_states"].extend(states.get("old_states", []))
                all_states["new_states"].extend(states.get("new_states", []))

            # Mark subtask as completed and advance
            self.user.mark_subtask_completed()
            self.user.advance_to_next_subtask()
            print(f"\n   ➡️  Subtask {i+1} complete, advancing to next...")
            print("=" * 60)

        # Phase 3: Aggregate results
        print("\n" + "=" * 80)
        print("📊 PHASE 3: AGGREGATING RESULTS")
        overall_duration = time_module.perf_counter() - overall_start
        aggregated_reward = self._aggregate_rewards(all_reward_infos)

        print(f"   Total duration: {overall_duration:.1f}s")
        print(f"   Total messages: {len(all_messages)}")
        print(f"   Subtask rewards:")
        for i, ri in enumerate(all_reward_infos):
            reward_val = ri.reward if ri else 'N/A'
            print(f"      Subtask {i}: {reward_val}")
        print(f"   Aggregated reward: {aggregated_reward.reward}")

        # Calculate costs
        res = get_cost(all_messages)
        if res is None:
            agent_cost, user_cost = None, None
        else:
            agent_cost, user_cost = res
        print(f"   Agent cost: {agent_cost}, User cost: {user_cost}")

        # Print final memory state
        final_memory = self.agent.memory.read()
        print(f"\n   🧠 Final memory state ({len(final_memory)} chars):")
        for line in final_memory.split("\n")[:15]:
            print(f"      {line}")
        print("=" * 80 + "\n")

        # Collect per-subtask memory snapshots into states
        memory_snapshots = {}
        for r in all_subtask_results:
            memory_snapshots[f"subtask_{r['subtask_idx']}_memory"] = r.get("memory_content", "")
        memory_snapshots["final_memory"] = self.agent.memory.read()
        all_states["memory_snapshots"] = memory_snapshots

        simulation_run = SimulationRun(
            id=str(uuid.uuid4()),
            task_id=self.task.id,
            start_time=overall_start_time,
            end_time=get_now(),
            duration=overall_duration,
            termination_reason=self._get_overall_termination_reason(all_subtask_results),
            reward_info=aggregated_reward,
            user_cost=user_cost,
            agent_cost=agent_cost,
            messages=all_messages,
            states=all_states,
            seed=self.seed,
        )

        logger.info(
            f"Personalization simulation complete. "
            f"Subtasks: {len(self.task.subtasks)}, "
            f"Reward: {aggregated_reward.reward if aggregated_reward else 'N/A'}"
        )

        return simulation_run

    def _setup_subtask_environment(self, subtask: SubTask) -> Environment:
        """Set up the environment for a subtask.

        Creates the domain environment, then injects memory tools from
        the agent's memory backend so the agent can actively query preferences.

        Args:
            subtask: The subtask to set up.

        Returns:
            Environment configured for the subtask's domain with memory tools.
        """
        from vita.registry import registry

        domain = subtask.domain
        env_data = subtask.environment

        if "," in domain:
            environment = get_cross_environment(domain, env_data, self.language)
        else:
            environment_constructor = registry.get_env_constructor(domain)
            environment = environment_constructor(env_data, self.language)

        # Inject memory tools into the domain environment
        self._inject_memory_tools(environment)

        return environment

    def _inject_memory_tools(self, environment: Environment):
        """Inject the agent's memory @is_tool methods into the domain environment.

        Memory tools (e.g., read_preference_memory, query_preference_memory,
        or any custom tools defined by the researcher) are discovered via the
        ToolKitBase metaclass and merged into the environment's toolkit so
        the agent can call them during conversation.

        Args:
            environment: The domain environment to inject tools into.
        """
        memory_tools = self.agent.memory.get_tools()
        if not memory_tools or environment.tools is None:
            return

        from vita.environment.toolkit import TOOL_ATTR

        for tool_name in memory_tools:
            if not hasattr(environment.tools, tool_name):
                # Bind the memory's tool method to the domain toolkit
                memory_method = getattr(self.agent.memory, tool_name)
                setattr(environment.tools, tool_name, memory_method)

    def _run_subtask(
        self,
        subtask: SubTask,
        environment: Environment,
        subtask_idx: int,
    ) -> Dict[str, Any]:
        """Run a single subtask conversation using the standard orchestrator loop.

        Args:
            subtask: The subtask to run.
            environment: The environment for this subtask.
            subtask_idx: Index of the subtask.

        Returns:
            Dictionary with subtask results including messages, states, termination reason.
        """
        # Create a temporary Task object for the orchestrator
        temp_task = Task(
            id=f"{self.task.id}_subtask_{subtask.subtask_id}",
            domain=subtask.domain,
            environment=subtask.environment,
            user_scenario=subtask.user_scenario or UserScenario(
                user_profile=self.task.user_profile
            ),
            instructions=subtask.instruction,
            evaluation_criteria=subtask.evaluation_criteria,
            message_history=subtask.message_history,
        )

        # Use the standard orchestrator for the inner conversation loop
        orchestrator = Orchestrator(
            domain=subtask.domain,
            agent=self.agent,
            user=self.user,
            environment=environment,
            task=temp_task,
            max_steps=self.max_steps_per_subtask,
            max_errors=self.max_errors,
            seed=self.seed,
            solo_mode=False,
            language=self.language,
        )

        simulation = orchestrator.run()

        return {
            "subtask_id": subtask.subtask_id,
            "subtask_idx": subtask_idx,
            "messages": simulation.messages,
            "states": simulation.states,
            "termination_reason": simulation.termination_reason,
            "duration": simulation.duration,
        }

    def _evaluate_subtask(
        self, subtask: SubTask, subtask_result: Dict[str, Any]
    ) -> Optional[RewardInfo]:
        """Evaluate a single subtask.

        Args:
            subtask: The subtask to evaluate.
            subtask_result: The result from running the subtask.

        Returns:
            RewardInfo for the subtask, or None.
        """
        # Create a temporary Task for evaluation
        temp_task = Task(
            id=f"{self.task.id}_subtask_{subtask.subtask_id}",
            domain=subtask.domain,
            environment=subtask.environment,
            user_scenario=subtask.user_scenario or UserScenario(
                user_profile=self.task.user_profile
            ),
            instructions=subtask.instruction,
            evaluation_criteria=subtask.evaluation_criteria,
        )

        # Create a temporary SimulationRun for evaluation
        temp_simulation = SimulationRun(
            id=str(uuid.uuid4()),
            task_id=temp_task.id,
            start_time=get_now(),
            end_time=get_now(),
            duration=subtask_result.get("duration", 0.0),
            termination_reason=subtask_result["termination_reason"],
            messages=subtask_result["messages"],
            states=subtask_result.get("states", {"old_states": [], "new_states": []}),
        )

        try:
            reward_info = evaluate_simulation(
                domain=subtask.domain,
                task=temp_task,
                simulation=temp_simulation,
                evaluation_type=self.evaluation_type,
                llm_evaluator=self.llm_evaluator,
                llm_args_evaluator=self.llm_args_evaluator,
                language=self.language,
            )
            logger.info(
                f"  Subtask {subtask.subtask_id} evaluation: reward={reward_info.reward}"
            )

            if self.enable_outcome_reward:
                reward_info = self._apply_outcome_reward(
                    subtask=subtask,
                    subtask_result=subtask_result,
                    reward_info=reward_info,
                )

            return reward_info
        except Exception as e:
            logger.error(
                f"  Error evaluating subtask {subtask.subtask_id}: {e}"
            )
            return RewardInfo(
                reward=0.0,
                info={"note": f"Evaluation error: {str(e)}"},
            )

    def _apply_outcome_reward(
        self,
        subtask: SubTask,
        subtask_result: Dict[str, Any],
        reward_info: RewardInfo,
    ) -> RewardInfo:
        """Combine trajectory reward with outcome (action) reward via min().

        If the subtask contains no WRITE tool calls, outcome is marked N/A and the
        trajectory reward is returned unchanged. Otherwise the judge LLM scores the
        last resolved action against the rubric and the subtask reward becomes
        min(traj_reward, action_reward).
        """
        from vita.evaluator.outcome_evaluator import evaluate_outcome_for_subtask

        rubrics: List[str] = []
        if subtask.evaluation_criteria and subtask.evaluation_criteria.overall_rubrics:
            rubrics = list(subtask.evaluation_criteria.overall_rubrics)

        try:
            outcome = evaluate_outcome_for_subtask(
                rubric=rubrics,
                env=subtask.environment,
                messages=subtask_result.get("messages", []),
                llm_evaluator=self.llm_evaluator,
                llm_args_evaluator=self.llm_args_evaluator,
            )
        except Exception as e:
            logger.error(f"  Outcome eval failed for subtask {subtask.subtask_id}: {e}")
            outcome = {
                "has_action": False,
                "action_reward": None,
                "action_type": None,
                "llm_result": {},
                "note": f"Outcome eval error: {e}",
            }

        info = dict(reward_info.info or {})
        info["traj_reward"] = reward_info.reward
        info["outcome_reward"] = outcome.get("action_reward")
        info["outcome_detail"] = outcome

        if outcome.get("has_action") and outcome.get("action_reward") is not None:
            new_reward = min(reward_info.reward, outcome["action_reward"])
            logger.info(
                f"  Subtask {subtask.subtask_id} outcome reward={outcome['action_reward']}, "
                f"combined={new_reward}"
            )
        else:
            new_reward = reward_info.reward

        return RewardInfo(
            reward=new_reward,
            nl_rubrics=reward_info.nl_rubrics,
            reward_breakdown=reward_info.reward_breakdown,
            info=info,
            window_evaluations=reward_info.window_evaluations,
        )

    def _aggregate_rewards(
        self, reward_infos: List[Optional[RewardInfo]]
    ) -> RewardInfo:
        """Aggregate rewards from all subtasks (average).

        Args:
            reward_infos: List of RewardInfo from each subtask.

        Returns:
            Aggregated RewardInfo.
        """
        valid_rewards = [ri for ri in reward_infos if ri is not None]
        if not valid_rewards:
            return RewardInfo(
                reward=0.0,
                info={"note": "No subtasks were evaluated"},
            )

        avg_reward = sum(ri.reward for ri in valid_rewards) / len(valid_rewards)

        # Build detailed breakdown
        breakdown = {}
        subtask_skill_tested = {}
        for i, ri in enumerate(reward_infos):
            if ri is not None:
                breakdown[f"subtask_{i}_reward"] = ri.reward
            # Record skill_tested for each subtask (regardless of evaluation)
            if i < len(self.task.subtasks):
                subtask_skill_tested[f"subtask_{i}"] = self.task.subtasks[i].skill_tested

        return RewardInfo(
            reward=avg_reward,
            info={
                "num_subtasks": len(self.task.subtasks),
                "num_evaluated": len(valid_rewards),
                "subtask_rewards": breakdown,
                "subtask_skill_tested": subtask_skill_tested,
                "aggregation": "average",
            },
        )

    def _get_overall_termination_reason(
        self, subtask_results: List[Dict[str, Any]]
    ) -> str:
        """Determine the overall termination reason.

        Args:
            subtask_results: Results from all subtasks.

        Returns:
            Termination reason string.
        """
        if not subtask_results:
            return TerminationReason.MAX_STEPS.value

        # If the last subtask ended normally, the overall simulation succeeded
        last_result = subtask_results[-1]
        last_reason = last_result.get("termination_reason")

        # Check if any subtask had a critical failure
        for result in subtask_results:
            reason = result.get("termination_reason")
            if reason in (
                TerminationReason.TOO_MANY_ERRORS.value,
                TerminationReason.TOO_MANY_ERRORS,
            ):
                return TerminationReason.TOO_MANY_ERRORS.value

        # Default: use the last subtask's termination reason
        if isinstance(last_reason, TerminationReason):
            return last_reason.value
        return last_reason or TerminationReason.AGENT_STOP.value
