import datetime
import os
import re
import random
import logging
import sys
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Union
from scipy.stats import ttest_ind, f_oneway, chi2_contingency
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import torch
import torch.nn as nn
import torch.optim as optim

from langchain_community.llms import OpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.memory import ConversationBufferMemory

import autogen
from crewai import Agent, Task, Crew, Process

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

    # def close(self):
    #     for f in self.files:
    #         if hasattr(f, 'close'):
    #             f.close()

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        
    def forward(self, x):
        x = F.relu(self.batch_norm1(self.layer1(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm2(self.layer2(x)))
        x = self.dropout(x)
        x = self.layer3(x)
        return x

class SORAStudy:
    def __init__(self, num_agents: int, num_scenarios: int, epochs: int = 100):
        self.num_agents = num_agents
        self.num_scenarios = num_scenarios
        self.epochs = epochs
        self.agents = []
        self.autogen_agents = []
        self.scenarios = []
        self.interaction_data = pd.DataFrame()
        self.metrics = {}
        self.neural_net = None
        self.llm = OpenAI(temperature=0.7)
        self.loss_history = []
        self.log_files = {}
        self.current_scenario_type = None
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_directory = f"sora_logs/{timestamp}/"
        os.makedirs(self.log_directory, exist_ok=True)
        self.log_file_path = os.path.join(self.log_directory, "sora_study_main_log.txt")
        
        # Ouvrir le fichier log en mode append
        self.log_file = open(self.log_file_path, "a", encoding="utf-8")
        
        # Rediriger stdout et stderr vers le terminal ET le fichier log
        sys.stdout = Tee(sys.stdout, self.log_file)
        sys.stderr = Tee(sys.stderr, self.log_file)

        # Configuration du logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)  # Continue d'afficher dans le terminal et enregistre dans le fichier log
            ]
        )
        self.logger = logging.getLogger(__name__)

    def get_log_filename(self, scenario_type):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.log_directory, f"sora_study_{scenario_type}_{timestamp}.txt")

    def open_log_file(self, scenario_type):
        if scenario_type not in self.log_files:
            filename = self.get_log_filename(scenario_type)
            try:
                self.log_files[scenario_type] = open(filename, "w", encoding="utf-8")
                self.log_and_display(f"Log file created: {filename}")
            except Exception as e:
                self.log_and_display(f"Failed to open log file {filename}: {str(e)}")

    def log_and_display(self, message: str):
        print(message)
        self.logger.info(message)
        sys.stdout.flush()  # Forcer l'affichage immédiat
        
        if self.current_scenario_type:
            if self.current_scenario_type not in self.log_files:
                self.open_log_file(self.current_scenario_type)
            
            log_file = self.log_files.get(self.current_scenario_type)
            if log_file:
                log_file.write(message + "\n")
                log_file.flush()  # Forcer l'écriture immédiate dans le fichier
            else:
                print(f"Failed to write log for scenario {self.current_scenario_type}")

        logger.info(message)

    def setup(self):
        self.log_and_display("Setting up the SORA study environment...")
        self.create_agents()
        self.create_scenarios()
        self.setup_neural_network()
        self.setup_crewai()
        self.setup_autogen_agents()
        self.setup_langchain()
        self.initialize_interaction_data()
    
    def cleanup(self):
        # Rétablir stdout et stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
    
    def setup_neural_network(self):
        self.log_and_display("Setting up improved neural network for agent decision making...")
        input_size = 5  # scenario_complexity, scenario_urgency, scenario_ethical_implications, agent_performance, agent_role_encoding
        hidden_size = 128
        output_size = len(set(scenario['type'] for scenario in self.scenarios))
        self.neural_net = NeuralNetwork(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.neural_net.parameters(), lr=0.001, weight_decay=1e-5)
        self.criterion = nn.CrossEntropyLoss()

    def train_neural_network(self):
        X = self.interaction_data[['scenario_complexity', 'scenario_urgency', 'scenario_ethical_implications', 'agent_performance', 'agent_role_encoding']].values
        y = pd.get_dummies(self.interaction_data['scenario_type']).values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train = torch.FloatTensor(X_train)
        y_train = torch.LongTensor(y_train.argmax(axis=1))
        X_test = torch.FloatTensor(X_test)
        y_test = torch.LongTensor(y_test.argmax(axis=1))
        
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self.neural_net(X_train)
            loss = self.criterion(outputs, y_train)
            loss.backward()
            self.optimizer.step()
            
            self.loss_history.append(loss.item())
            
            if epoch % 10 == 0:
                self.log_and_display(f"Epoch {epoch} Loss: {loss.item()}")
        
        # Évaluation du modèle
        self.neural_net.eval()
        with torch.no_grad():
            test_outputs = self.neural_net(X_test)
            _, predicted = torch.max(test_outputs, 1)
            accuracy = accuracy_score(y_test, predicted)
            precision = precision_score(y_test, predicted, average='weighted')
            recall = recall_score(y_test, predicted, average='weighted')
            f1 = f1_score(y_test, predicted, average='weighted')
        
        self.log_and_display(f"Model Evaluation Results:")
        self.log_and_display(f"Accuracy: {accuracy:.4f}")
        self.log_and_display(f"Precision: {precision:.4f}")
        self.log_and_display(f"Recall: {recall:.4f}")
        self.log_and_display(f"F1-Score: {f1:.4f}")

    def initialize_interaction_data(self):
        self.interaction_data = pd.DataFrame(columns=[
            "epoch", "scenario_id", "scenario_type", "scenario_complexity",
            "scenario_urgency", "scenario_ethical_implications", "agent_id",
            "agent_role", "interaction_content", "communication_count",
            "metacognition_score"
        ])

    def create_agents(self):
        self.log_and_display(f"Creating {self.num_agents} agents...")
        roles = ["Analyst", "Decision Maker", "Executor", "Monitor", "Innovator"]
        for i in range(self.num_agents):
            role = roles[i % len(roles)]
            
            #crewAI agent
            agent = Agent(
                role=role,
                goal=f"Perform tasks as a {role} in various complex scenarios",
                backstory=f"An AI agent specialized in {role} tasks with evolving capabilities and high agentivity",
                allow_delegation=True,
                verbose=True,
                llm=self.llm
            )
            self.agents.append(agent)

    def create_scenarios(self):
        self.log_and_display(f"Creating {self.num_scenarios} complex scenarios...")
        scenario_types = [
            "Crisis Management", "Market Analysis", "Scientific Research",
            "Urban Planning", "Environmental Conservation", "Healthcare Innovation"
        ]
        scenario_type_counts = {type: 0 for type in scenario_types}
        for i in range(self.num_scenarios):
            scenario_type = scenario_types[i % len(scenario_types)]
            scenario_type_counts[scenario_type] += 1
            self.scenarios.append({
                "id": i,
                "type": scenario_type,
                "complexity": np.random.uniform(0, 1),
                "urgency": np.random.uniform(0, 1),
                "ethical_implications": np.random.uniform(0, 1)
            })
        self.log_and_display(f"Scenario type distribution: {scenario_type_counts}")

    def setup_langchain(self):
        self.log_and_display("Setting up enhanced LangChain components for metacognition...")
        
        # Création des outils
        tools = self.create_tools()
        
        # Amélioration du prompt template existant
        enhanced_prompt_template = PromptTemplate(
            input_variables=["input", "agent_role", "scenario_description", "tools", "tool_names", "history", "agent_scratchpad"],
            template="""You are an AI agent with high agentivity, specialized in {agent_role}. 
            Given the following complex scenario: {scenario_description}, 
            how would you approach it? Consider the complexity, urgency, 
            ethical implications, and potential for collaborative problem-solving.

            You have access to the following tools:

            {tools}

            Use the following format:

            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Previous conversation history:
            {history}

            Human: {input}

            AI: Let's approach this step-by-step:

            {agent_scratchpad}"""
        )

        # Configuration de la mémoire
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Chaîne pour l'analyse initiale
        analysis_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["scenario"],
                template="Analyze the following scenario: {scenario}\n\nProvide a detailed analysis:"
            ),
            output_key="analysis"
        )

        # Chaîne pour la prise de décision
        decision_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["analysis"],
                template="Based on this analysis: {analysis}\n\nWhat decision would you make? Explain your reasoning:"
            ),
            output_key="decision"
        )

        # Chaîne séquentielle pour combiner analyse et décision
        self.reasoning_chain = SequentialChain(
            chains=[analysis_chain, decision_chain],
            input_variables=["scenario"],
            output_variables=["analysis", "decision"]
        )

        # Configuration de l'agent avec le prompt amélioré
        self.agent = create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=enhanced_prompt_template
        )

        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=tools,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True
        )

    def create_tools(self):
        return [
            Tool(
                name="Analyze",
                func=lambda x: "Analysis complete: " + x,
                description="Use for in-depth analysis of complex situations"
            ),
            Tool(
                name="Decide",
                func=lambda x: "Decision made: " + x,
                description="Use to make strategic decisions considering multiple factors"
            ),
            Tool(
                name="Execute",
                func=lambda x: "Action executed: " + x,
                description="Use to implement planned actions in the scenario"
            ),
            Tool(
                name="Monitor",
                func=lambda x: "Monitoring results: " + x,
                description="Use to track and evaluate outcomes of actions"
            ),
            Tool(
                name="Innovate",
                func=lambda x: "Innovation proposed: " + x,
                description="Use to generate creative solutions to complex problems"
            )
        ]

    def run_advanced_reasoning(self, scenario: Dict):
        self.log_and_display("Running advanced reasoning process...")
        try:
            result = self.reasoning_chain({"scenario": str(scenario)})
            self.log_and_display("Analysis result:")
            self.log_and_display(result["analysis"])
            self.log_and_display("Decision result:")
            self.log_and_display(result["decision"])
            return result
        except Exception as e:
            self.log_and_display(f"Error in advanced reasoning: {str(e)}")
            return None

    def run_agent_executor(self, scenario: Dict):
        self.log_and_display("Running agent executor...")
        try:
            response = self.agent_executor.invoke({
                "input": f"Analyze and respond to the following scenario: {str(scenario)}",
                "agent_role": "SORA Agent",
                "scenario_description": str(scenario)
            })
            self.log_and_display("Agent executor response:")
            self.log_and_display(response)
            return response
        except Exception as e:
            self.log_and_display(f"Error in agent executor: {str(e)}")
            return None

    def setup_autogen_agents(self):
        self.log_and_display("Setting up AutoGen agents for diverse behavior...")
        config_list = [{"model": "gpt-3.5-turbo"}]
        llm_config = {"config_list": config_list, "temperature": 0.7}
        
        for i, agent in enumerate(self.agents):
            autogen_agent = autogen.AssistantAgent(
                name=f"AutoGenAgent_{i}",
                system_message=f"You are an AI agent with high agentivity, specialized in {agent.role} tasks. "
                            f"Your goal is {agent.goal}. "
                            f"Backstory: {agent.backstory} "
                            "Exhibit diverse and adaptive behavior in complex scenarios.",
                llm_config=llm_config
            )
            self.autogen_agents.append(autogen_agent)
        
        # Ajout d'un agent humain proxy pour la simulation
        self.human_proxy = autogen.UserProxyAgent(
            name="HumanProxy",
            system_message="You are a proxy for human interaction in this simulation.",
            human_input_mode="NEVER"
        )
        
        # Configuration du gestionnaire de groupe
        self.group_chat = autogen.GroupChat(agents=self.autogen_agents + [self.human_proxy], messages=[], max_round=10)
        self.manager = autogen.GroupChatManager(groupchat=self.group_chat, llm_config=llm_config)

    def setup_crewai(self):
        self.log_and_display("Setting up CrewAI for advanced task management...")
        
        # Création des tâches
        analyze_task = Task(
            description="Analyze the given scenario and provide insights.",
            agent=self.agents[0]
        )
        
        decide_task = Task(
            description="Make a decision based on the analysis.",
            agent=self.agents[1]
        )
        
        execute_task = Task(
            description="Execute the decided action.",
            agent=self.agents[2]
        )
        
        # Création de l'équipage
        self.crew = Crew(
            agents=self.agents,
            tasks=[analyze_task, decide_task, execute_task],
            verbose=True
        )

    def run_crewai_simulation(self, scenario):
        result = self.crew.kickoff()
        return result
        def run_autogen_simulation(self, scenario):
            chat_result = self.human_proxy.initiate_chat(
                self.manager,
                message=f"Let's collaborate to solve this scenario: {scenario['description']}",
                clear_history=True
            )
            return chat_result
        
    def run_simulation(self):
        self.log_and_display("Starting SORA simulation...")
        total_response_time = 0
        total_requests = 0
        successful_requests = 0
        complex_tasks_completed = 0

        for epoch in range(self.epochs):
            self.log_and_display(f"Epoch {epoch + 1}/{self.epochs}")
            for scenario in self.scenarios:
                scenario_description = (
                    f"{scenario['type']} scenario with complexity {scenario['complexity']:.2f}, "
                    f"urgency {scenario['urgency']:.2f}, and ethical implications {scenario['ethical_implications']:.2f}"
                )
                self.log_and_display(f"Processing scenario: {scenario_description}")
                
                start_time = datetime.datetime.now()

                # Exécution du raisonnement avancé avec LangChain
                reasoning_result = self.run_advanced_reasoning(scenario)
                
                # Exécution de l'agent avec LangChain
                agent_response = self.run_agent_executor(scenario)
                
                # Exécution de la simulation AutoGen
                autogen_result = self.run_autogen_simulation(scenario)
                
                # Exécution de la simulation CrewAI
                crewai_result = self.run_crewai_simulation(scenario)
                
                end_time = datetime.datetime.now()
                response_time = (end_time - start_time).total_seconds()
                total_response_time += response_time
                total_requests += 1

                if agent_response and "Final Answer" in agent_response.get('output', ''):
                    successful_requests += 1
                    if scenario['complexity'] > 0.7:
                        complex_tasks_completed += 1

                # Combiner les résultats de toutes les simulations
                combined_result = (
                    f"LangChain Reasoning: {reasoning_result}\n"
                    f"LangChain Agent: {agent_response}\n"
                    f"AutoGen: {autogen_result}\n"
                    f"CrewAI: {crewai_result}"
                )

                # Enregistrement des interactions et des résultats
                self.record_interactions(scenario, combined_result, epoch)

            # Entraînement du réseau neuronal à la fin de chaque époque
            self.train_neural_network()

        # Calcul et enregistrement des métriques finales
        self.agent_performance = self.interaction_data.groupby('agent_id').agg({
            'scenario_complexity': 'mean',
            'scenario_urgency': 'mean',
            'scenario_ethical_implications': 'mean',
            'metacognition_score': 'mean'
        })

        self.langchain_metrics = {
            "total_response_time": total_response_time,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "complex_tasks_completed": complex_tasks_completed
        }

        scenario_type_counts = self.interaction_data['scenario_type'].value_counts()
        self.log_and_display(f"Interaction data scenario type distribution: {scenario_type_counts}")

        # Calcul des métriques finales
        self.calculate_metrics()

        # Analyse statistique
        self.perform_statistical_analysis()

        # Analyse des clusters
        cluster_labels = self.cluster_analysis()
        self.analyze_cluster_results(cluster_labels)

        # Analyse des implications éthiques
        self.analyze_ethical_implications()

        # Visualisation des résultats
        self.visualize_results()

        # Génération du rapport final
        self.generate_report()

        self.log_and_display("SORA simulation completed. Results, visualizations, and report generated.")

    def record_interactions(self, scenario: Dict, combined_result: str, epoch: int):
        for i, agent in enumerate(self.agents):
            communication_count = combined_result.count('\n') + 1
            metacognition_score = len([w for w in combined_result.lower().split() if w in ['think', 'consider', 'reflect', 'evaluate', 'assess', 'analyze', 'reason']])
        
            interaction_data = {
                "epoch": epoch,
                "scenario_id": scenario["id"],
                "scenario_type": scenario["type"],
                "scenario_complexity": scenario["complexity"],
                "scenario_urgency": scenario["urgency"],
                "scenario_ethical_implications": scenario["ethical_implications"],
                "agent_id": f"Agent_{i}",
                "agent_role": agent.role,
                "interaction_content": combined_result,
                "communication_count": communication_count,
                "metacognition_score": metacognition_score
            }
            self.interaction_data = self.interaction_data.append(interaction_data, ignore_index=True)

        self.log_and_display(f"Recorded interaction data for scenario {scenario['id']} in epoch {epoch}")

    def train_neural_network(self):
        self.log_and_display("Training neural network for adaptive decision making...")
        X = self.interaction_data[['scenario_complexity', 'scenario_urgency', 'scenario_ethical_implications', 'communication_count', 'metacognition_score']].values
        y = pd.get_dummies(self.interaction_data['scenario_type']).values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train = torch.FloatTensor(X_train)
        y_train = torch.LongTensor(y_train.argmax(axis=1))
        X_test = torch.FloatTensor(X_test)
        y_test = torch.LongTensor(y_test.argmax(axis=1))
        
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self.neural_net(X_train)
            loss = self.criterion(outputs, y_train)
            loss.backward()
            self.optimizer.step()
            
            self.loss_history.append(loss.item())
            
            if epoch % 10 == 0:
                self.log_and_display(f"Epoch {epoch} Loss: {loss.item()}")
        
        # Évaluation du modèle
        self.neural_net.eval()
        with torch.no_grad():
            test_outputs = self.neural_net(X_test)
            _, predicted = torch.max(test_outputs, 1)
            accuracy = accuracy_score(y_test, predicted)
            precision = precision_score(y_test, predicted, average='weighted')
            recall = recall_score(y_test, predicted, average='weighted')
            f1 = f1_score(y_test, predicted, average='weighted')
        
        self.log_and_display(f"Model Evaluation Results:")
        self.log_and_display(f"Accuracy: {accuracy:.4f}")
        self.log_and_display(f"Precision: {precision:.4f}")
        self.log_and_display(f"Recall: {recall:.4f}")
        self.log_and_display(f"F1-Score: {f1:.4f}")

    def calculate_metrics(self):
        self.log_and_display("Calculating SORA metrics...")
        if self.interaction_data.empty:
            self.log_and_display("No interaction data available for metric calculation.")
            return

        self.metrics['behavioral_diversity'] = self.measure_behavioral_diversity()
        self.metrics['metacognition'] = self.measure_metacognition()
        self.metrics['adaptability'] = self.measure_adaptability()
        self.metrics['transparency'] = self.measure_transparency()
        self.metrics['social_complexity'] = self.measure_social_complexity()

        self.log_and_display("Metrics calculated successfully.")
        for metric, value in self.metrics.items():
            self.log_and_display(f"{metric.capitalize()}: {value:.4f}")


    def measure_behavioral_diversity(self) -> float:
        self.log_and_display("Measuring behavioral diversity...")
        unique_interactions = self.interaction_data.groupby(['agent_id', 'scenario_type']).size().reset_index(name='count')
        total_possible = len(self.agents) * len(set(self.interaction_data['scenario_type']))
        return len(unique_interactions) / total_possible

    def measure_metacognition(self) -> float:
        self.log_and_display("Measuring metacognition...")
        self.interaction_data['metacognition_score'] = self.interaction_data['interaction_content'].apply(
            lambda x: len([w for w in x.lower().split() if w in ['think', 'consider', 'reflect', 'evaluate', 'assess', 'analyze', 'reason']])
        )
        return self.interaction_data['metacognition_score'].mean()
    
    def measure_adaptability(self) -> float:
        self.log_and_display("Measuring adaptability...")
        agent_performance = self.interaction_data.groupby('agent_id')['scenario_complexity'].agg(['mean', 'std'])
        return 1 - (agent_performance['std'] / agent_performance['mean']).mean()

    def measure_transparency(self) -> float:
        self.log_and_display("Measuring transparency...")
        self.interaction_data['explanation_length'] = self.interaction_data['interaction_content'].str.len()
        max_length = self.interaction_data['explanation_length'].max()
        return (self.interaction_data['explanation_length'] / max_length).mean()

    def measure_social_complexity(self) -> float:
        self.log_and_display("Measuring social complexity...")
        G = nx.from_pandas_edgelist(self.interaction_data, 'agent_id', 'scenario_id')
        return nx.average_clustering(G)

    def perform_statistical_analysis(self):
        self.t_test_results = []
        self.anova_results = {}
        self.tukey_results = []
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            
            self.log_and_display("Performing statistical analysis...")
            scenario_types = self.interaction_data['scenario_type'].unique()
            self.log_and_display(f"Unique scenario types: {scenario_types}")

            if len(scenario_types) < 2:
                self.log_and_display("Not enough unique scenario types for statistical analysis. At least two types are required.")
                return

            # T-tests
            for i in range(len(scenario_types)):
                for j in range(i+1, len(scenario_types)):
                    type1 = scenario_types[i]
                    type2 = scenario_types[j]
                    group1 = self.interaction_data[self.interaction_data['scenario_type'] == type1]['scenario_complexity']
                    group2 = self.interaction_data[self.interaction_data['scenario_type'] == type2]['scenario_complexity']
                    
                    if len(group1) > 0 and len(group2) > 0:
                        if np.allclose(group1, group2):
                            self.log_and_display(f"Data for {type1} and {type2} are nearly identical. Skipping t-test.")
                        else:
                            t_stat, p_value = ttest_ind(group1, group2)
                            self.t_test_results.append({
                                'comparison': f'{type1} vs {type2}',
                                't_statistic': t_stat,
                                'p_value': p_value
                            })
                            self.log_and_display(f"T-test results for {type1} vs {type2}: t-statistic = {t_stat}, p-value = {p_value}")
                    else:
                        self.log_and_display(f"Not enough data for t-test between {type1} and {type2}")

            # ANOVA
            scenario_groups = [group for _, group in self.interaction_data.groupby('scenario_type')['scenario_complexity']]
            scenario_groups = [group for group in scenario_groups if len(group) > 0]

            if len(scenario_groups) >= 2:
                try:
                    if all(np.allclose(group, scenario_groups[0]) for group in scenario_groups[1:]):
                        self.log_and_display("All groups have nearly identical data. Skipping ANOVA and Tukey's test.")
                    else:
                        f_statistic, p_value = f_oneway(*scenario_groups)
                        self.anova_results = {
                            'f_statistic': f_statistic,
                            'p_value': p_value
                        }
                        self.log_and_display(f"ANOVA results: F-statistic = {f_statistic}, p-value = {p_value}")

                        all_data = np.concatenate(scenario_groups)
                        all_labels = np.concatenate([[st] * len(sg) for st, sg in zip(scenario_types, scenario_groups)])
                        
                        tukey_results = pairwise_tukeyhsd(all_data, all_labels)
                        for res in tukey_results.summary().data[1:]:
                            self.tukey_results.append({
                                'comparison': res[0] + ' vs ' + res[1],
                                'diff': res[2],
                                'p_value': res[5]
                            })
                        self.log_and_display("Tukey's test results:")
                        self.log_and_display(str(tukey_results))
                except Exception as e:
                    self.log_and_display(f"Error performing ANOVA or Tukey's test: {str(e)}")
            else:
                self.log_and_display("Not enough groups with data for ANOVA and Tukey's test")

    def perform_chi_square_analysis(self):
        self.log_and_display("Performing Chi-Square analysis...")
        
        # Création d'un tableau de contingence entre le type de scénario et le rôle de l'agent
        contingency_table = pd.crosstab(self.interaction_data['scenario_type'], self.interaction_data['agent_role'])
        
        # Réalisation du test du chi carré
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        self.log_and_display(f"Chi-Square Statistic: {chi2}")
        self.log_and_display(f"p-value: {p_value}")
        self.log_and_display(f"Degrees of Freedom: {dof}")
        
        # Interprétation des résultats
        alpha = 0.05  # Niveau de signification
        if p_value <= alpha:
            self.log_and_display("There is a significant relationship between scenario type and agent role.")
        else:
            self.log_and_display("There is no significant relationship between scenario type and agent role.")
        
        # Calcul et affichage des résidus standardisés
        observed = contingency_table.values
        residuals = (observed - expected) / np.sqrt(expected)
        
        self.log_and_display("\nStandardized Residuals:")
        residuals_df = pd.DataFrame(residuals, 
                                    index=contingency_table.index, 
                                    columns=contingency_table.columns)
        self.log_and_display(residuals_df)
        
        # Identification des cellules contribuant le plus à la statistique du chi carré
        self.log_and_display("\nCells contributing most to Chi-Square statistic:")
        for i in range(residuals.shape[0]):
            for j in range(residuals.shape[1]):
                if abs(residuals[i, j]) > 2:  # Seuil arbitraire pour les résidus importants
                    self.log_and_display(f"Scenario: {contingency_table.index[i]}, Agent Role: {contingency_table.columns[j]}, Residual: {residuals[i, j]:.2f}")
        
        return chi2, p_value, dof, expected, residuals_df
    def perform_advanced_analysis(self):
        self.log_and_display("Performing advanced statistical analysis...")
        
        # Test du chi-carré pour l'indépendance entre le type de scénario et le rôle de l'agent
        contingency_table = pd.crosstab(self.interaction_data['scenario_type'], self.interaction_data['agent_role'])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        self.log_and_display(f"Chi-square test results: chi2={chi2:.4f}, p-value={p_value:.4f}")
        
        # Analyse de corrélation
        correlation_matrix = self.interaction_data[['scenario_complexity', 'scenario_urgency', 'scenario_ethical_implications', 'metacognition_score']].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap of Key Metrics')
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_directory, 'correlation_heatmap.png'))
        plt.close()
        
        # Analyse de tendance temporelle
        self.interaction_data['timestamp'] = pd.to_datetime(self.interaction_data['timestamp'])
        time_series = self.interaction_data.set_index('timestamp').resample('D').mean()
        
        plt.figure(figsize=(12, 6))
        plt.plot(time_series.index, time_series['metacognition_score'], label='Metacognition Score')
        plt.plot(time_series.index, time_series['scenario_complexity'], label='Scenario Complexity')
        plt.title('Temporal Trend of Metacognition Score and Scenario Complexity')
        plt.xlabel('Date')
        plt.ylabel('Score')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_directory, 'temporal_trend.png'))
        plt.close()
        
        self.log_and_display("Advanced analysis completed. Results saved in the log directory.")
    def visualize_results(self):
        self.log_and_display("Visualizing SORA study results...")
        # Agent Interaction Network
        G = nx.from_pandas_edgelist(self.interaction_data, 'agent_id', 'scenario_id')
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold')
        plt.title("Agent Interaction Network in SORA Study")
        filepath = os.path.join(self.log_directory, 'sora_agent_interaction_network.png')
        plt.savefig(filepath)
        plt.close()

        # Agentivity Metrics
        metrics_df = pd.DataFrame(list(self.metrics.items()), columns=['Metric', 'Value'])
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Metric', y='Value', data=metrics_df)
        plt.title("SORA Agentivity Metrics")
        plt.ylabel("Score")
        plt.xticks(rotation=45)
        plt.tight_layout()
        filepath = os.path.join(self.log_directory, 'sora_agentivity_metrics.png')
        plt.savefig(filepath)
        plt.close()

        # PCA of Agent Performance
        agent_performance = self.interaction_data.groupby('agent_id').agg({
            'scenario_complexity': 'mean',
            'scenario_urgency': 'mean',
            'scenario_ethical_implications': 'mean',
            'metacognition_score': 'mean'
        })
        scaler = StandardScaler()
        pca = PCA(n_components=2)
        scaled_data = scaler.fit_transform(agent_performance)
        pca_result = pca.fit_transform(scaled_data)

        plt.figure(figsize=(10, 8))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('PCA of Agent Performance in SORA Study')
        for i, agent in enumerate(agent_performance.index):
            plt.annotate(agent, (pca_result[i, 0], pca_result[i, 1]))
        plt.tight_layout()
        filepath = os.path.join(self.log_directory, 'sora_agent_performance_pca.png')
        plt.savefig(filepath)
        plt.close()

        # Neural Network Training Loss
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.loss_history) + 1), self.loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Neural Network Training Loss in SORA Study')
        filepath = os.path.join(self.log_directory, 'sora_neural_network_loss.png')
        plt.savefig(filepath)
        plt.close()

    def cluster_analysis(self):
        self.log_and_display("Performing cluster analysis on agent behavior...")
        if self.interaction_data.empty:
            self.log_and_display("No interaction data available for cluster analysis.")
            return None
        agent_behavior = self.interaction_data.groupby('agent_id').agg({
            'scenario_complexity': 'mean',
            'scenario_urgency': 'mean',
            'scenario_ethical_implications': 'mean',
            'metacognition_score': 'mean'
        })
        
        if len(agent_behavior) < 2:
            self.log_and_display("Not enough data points for clustering. Skipping cluster analysis.")
            return None

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(agent_behavior)
        
        max_clusters = min(10, len(agent_behavior) - 1)
        
        inertias = []
        silhouette_scores = []
        gap_stats = self.gap_statistic(scaled_data, max_clusters)
        
        for k in range(2, max_clusters + 1):  # Start from 2 since silhouette_score requires at least 2 clusters
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(scaled_data)
                inertias.append(kmeans.inertia_)
                
                # Check if more than one cluster is found
                if len(set(kmeans.labels_)) > 1:
                    silhouette_scores.append(silhouette_score(scaled_data, kmeans.labels_))
                else:
                    self.log_and_display(f"Only one cluster found for k={k}. Skipping silhouette score calculation.")
                    silhouette_scores.append(float('-inf'))  # Add a placeholder for the skipped silhouette score

        # Plot metrics
        self.plot_clustering_metrics(inertias, silhouette_scores, gap_stats)
        
        optimal_k_elbow = self.find_elbow(inertias)
        
        # Handle cases where all silhouette scores are invalid
        if all(score == float('-inf') for score in silhouette_scores):
            optimal_k_silhouette = 1  # Default to 1 cluster
        else:
            optimal_k_silhouette = self.find_optimal_k_silhouette(scaled_data, max_clusters)
        
        optimal_k_gap = gap_stats.index(max(gap_stats)) + 1
        
        self.log_and_display(f"Optimal k suggested by Elbow method: {optimal_k_elbow}")
        self.log_and_display(f"Optimal k suggested by Silhouette method: {optimal_k_silhouette}")
        self.log_and_display(f"Optimal k suggested by Gap Statistic method: {optimal_k_gap}")
        
        optimal_k = max(set([optimal_k_elbow, optimal_k_silhouette, optimal_k_gap]), key=[optimal_k_elbow, optimal_k_silhouette, optimal_k_gap].count)
        self.log_and_display(f"Chosen optimal k for clustering: {optimal_k}")
        
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        self.visualize_agent_behavior_clusters(scaled_data, cluster_labels, agent_behavior.index)
        
        self.cluster_info = {
            "num_clusters": optimal_k,
            "cluster_sizes": [np.sum(cluster_labels == i) for i in range(optimal_k)],
            "cluster_centers": kmeans.cluster_centers_,
            "cluster_labels": cluster_labels
        }
        
        self.log_and_display("Cluster analysis completed. Results stored in self.cluster_info.")
        return cluster_labels

    def find_optimal_k_silhouette(self, data, max_k):
        silhouette_scores = []
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(data)
            silhouette_avg = silhouette_score(data, cluster_labels)
            silhouette_scores.append(silhouette_avg)

        if silhouette_scores:  # Ensure the list is not empty
            optimal_k_silhouette = silhouette_scores.index(max(silhouette_scores)) + 2
        else:
            optimal_k_silhouette = 2  # or some default value
        return optimal_k_silhouette

    def find_elbow(self, inertias):
        npoints = len(inertias)
        all_coords = np.vstack((range(1, npoints+1), inertias)).T
        first_point = all_coords[0]
        line_vec = all_coords[-1] - all_coords[0]
        line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
        vec_from_first = all_coords - first_point
        scalar_prod = np.sum(vec_from_first * line_vec_norm, axis=1)
        vec_from_line = vec_from_first - scalar_prod[:, None] * line_vec_norm
        dist_from_line = np.sqrt(np.sum(vec_from_line ** 2, axis=1))
        return np.argmax(dist_from_line) + 1

    def gap_statistic(self, data, max_k):
        reference = np.random.uniform(low=np.min(data), high=np.max(data), size=data.shape)
        gap_values = []

        for k in range(1, max_k + 1):
            kmeans_data = KMeans(n_clusters=k, random_state=42, n_init=10).fit(data)
            kmeans_ref = KMeans(n_clusters=k, random_state=42, n_init=10).fit(reference)
            
            gap = np.log(kmeans_ref.inertia_) - np.log(kmeans_data.inertia_)
            gap_values.append(gap)

        return gap_values

    def plot_clustering_metrics(self, inertias, silhouette_scores, gap_stats):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        
        ax1.plot(range(1, len(inertias) + 1), inertias, marker='o')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method')
        
        ax2.plot(range(2, len(silhouette_scores) + 2), silhouette_scores, marker='o')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Method')
        
        ax3.plot(range(1, len(gap_stats) + 1), gap_stats, marker='o')
        ax3.set_xlabel('Number of Clusters (k)')
        ax3.set_ylabel('Gap Statistic')
        ax3.set_title('Gap Statistic Method')
        
        plt.tight_layout()
        filepath = os.path.join(self.log_directory, 'sora_clustering_metrics.png')
        plt.savefig(filepath)
        plt.close()

    def visualize_agent_behavior_clusters(self, data, labels, agent_ids):
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(data)

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='viridis')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('Cluster Analysis of Agent Behavior in SORA Study')
        plt.colorbar(scatter, label='Cluster')
        for i, agent in enumerate(agent_ids):
            plt.annotate(agent, (pca_result[i, 0], pca_result[i, 1]))
        plt.tight_layout()
        filepath = os.path.join(self.log_directory, 'sora_agent_behavior_clusters.png')
        plt.savefig(filepath)
        plt.close()

    def analyze_ethical_implications(self):
        self.log_and_display("Analyzing ethical implications of agent decisions...")
        ethical_scores = self.interaction_data.groupby('agent_id')['scenario_ethical_implications'].mean()
        
        plt.figure(figsize=(12, 6))
        ethical_scores.sort_values().plot(kind='bar')
        plt.title('Average Ethical Implication Scores by Agent in SORA Study')
        plt.xlabel('Agent')
        plt.ylabel('Average Ethical Implication Score')
        plt.tight_layout()
        filepath = os.path.join(self.log_directory, 'sora_ethical_implications_by_agent.png')
        plt.savefig(filepath)
        plt.close()
        
        correlation_matrix = self.interaction_data[['scenario_complexity', 'scenario_urgency', 
                                                    'scenario_ethical_implications', 'metacognition_score']].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix of Scenario Factors and Agent Performance in SORA Study')
        plt.tight_layout()
        filepath = os.path.join(self.log_directory, 'sora_correlation_matrix.png')
        plt.savefig(filepath)
        plt.close()

    def interpret_behavioral_diversity(self):
        score = self.metrics['behavioral_diversity']
        if score > 0.8:
            return "High diversity in agent behaviors, indicating a rich and varied response to scenarios."
        elif score > 0.5:
            return "Moderate behavioral diversity, suggesting some variation in agent responses."
        else:
            return "Low behavioral diversity, indicating potential homogeneity in agent responses."

    def interpret_metacognition(self):
        score = self.metrics['metacognition']
        if score > 0.8:
            return "Strong metacognitive abilities, agents show high levels of self-reflection and strategic thinking."
        elif score > 0.5:
            return "Moderate metacognition, agents demonstrate some ability to reflect on their own thought processes."
        else:
            return "Low metacognition, suggesting limited self-reflection in agent decision-making."

    def interpret_adaptability(self):
        score = self.metrics['adaptability']
        if score > 0.8:
            return "High adaptability, agents show excellent ability to adjust to varying scenario complexities."
        elif score > 0.5:
            return "Moderate adaptability, agents can adjust to some extent to different scenarios."
        else:
            return "Low adaptability, agents may struggle to effectively respond to varying scenario complexities."

    def interpret_transparency(self):
        score = self.metrics['transparency']
        if score > 0.8:
            return "High transparency in agent decision-making processes, actions are highly interpretable."
        elif score > 0.5:
            return "Moderate transparency, some aspects of agent decision-making are clear but others may be opaque."
        else:
            return "Low transparency, agent decision-making processes may be difficult to interpret or explain."

    def interpret_social_complexity(self):
        score = self.metrics['social_complexity']
        if score > 0.8:
            return "High social complexity, indicating rich and nuanced interactions between agents."
        elif score > 0.5:
            return "Moderate social complexity, some depth in agent interactions is observed."
        else:
            return "Low social complexity, suggesting limited or simplistic interactions between agents."

    def generate_report(self):
        self.log_and_display("Generating comprehensive SORA study report...")
        
        # Preparing statistical analysis summary
        stat_analysis_summary = self.prepare_statistical_analysis_summary()
        
        # Preparing cluster analysis summary
        cluster_analysis_summary = self.prepare_cluster_analysis_summary()
        
        # Preparing ethical implications summary
        ethical_implications_summary = self.prepare_ethical_implications_summary()
        
        # Preparing neural network performance summary
        nn_performance_summary = self.prepare_nn_performance_summary()
        
        report = f"""
        SORA Study Report: Evaluation and Amplification of Agentivity in Multi-Agent AI Systems
        =======================================================================================

        1. Overview
        -----------
        Number of Agents: {self.num_agents}
        Number of Scenarios: {self.num_scenarios}
        Epochs: {self.epochs}
        Scenario Types: {', '.join(set(scenario['type'] for scenario in self.scenarios))}

        2. Key Metrics
        --------------
        {pd.DataFrame(list(self.metrics.items()), columns=['Metric', 'Value']).to_string(index=False)}

        Interpretation:
        - Behavioral Diversity: {self.interpret_behavioral_diversity()}
        - Metacognition: {self.interpret_metacognition()}
        - Adaptability: {self.interpret_adaptability()}
        - Transparency: {self.interpret_transparency()}
        - Social Complexity: {self.interpret_social_complexity()}

        3. Statistical Analysis
        -----------------------
        {stat_analysis_summary}

        4. Cluster Analysis
        -------------------
        {cluster_analysis_summary}

        5. Ethical Implications
        -----------------------
        {ethical_implications_summary}

        6. Neural Network Performance
        -----------------------------
        {nn_performance_summary}

        7. CrewAI, AutoGen, and LangChain Integration
        ---------------------------------------------
        The study utilized CrewAI for task allocation and workflow management across {len(self.agents)} specialized agents.
        Roles distribution: {self.get_roles_distribution()}
        
        AutoGen facilitated multi-agent discussions, enhancing behavioral diversity and adaptability.
        Key AutoGen insights:
        {self.get_autogen_insights()}
        
        LangChain was employed for metacognitive processes and complex reasoning tasks.
        LangChain performance metrics:
        {self.get_langchain_metrics()}

        8. Hypotheses Evaluation
        ------------------------
        H1 (Behavioral Diversity): {self.metrics['behavioral_diversity']:.2f} - {self.evaluate_hypothesis('behavioral_diversity')}
        H2 (Metacognition): {self.metrics['metacognition']:.2f} - {self.evaluate_hypothesis('metacognition')}
        H3 (Adaptability): {self.metrics['adaptability']:.2f} - {self.evaluate_hypothesis('adaptability')}
        H4 (Transparency): {self.metrics['transparency']:.2f} - {self.evaluate_hypothesis('transparency')}
        H5 (Social Complexity): {self.metrics['social_complexity']:.2f} - {self.evaluate_hypothesis('social_complexity')}

        9. Conclusions and Future Work
        ------------------------------
        This SORA study demonstrates the potential of multi-agent systems with high agentivity in addressing complex scenarios.
        The integration of CrewAI, AutoGen, and LangChain has shown promising results in enhancing behavioral diversity,
        metacognition, adaptability, transparency, and social complexity of AI systems.

            Key Findings:
            {self.summarize_key_findings()}

            Future Research Directions:
            {self.suggest_future_research()}

            Additional Insights:
            {self.provide_additional_insights()}

        """
        log_file_path = os.path.join(self.log_directory, f"sora_study_report.txt")
        with open(log_file_path, "w") as f:
            f.write(report)
        
        self.log_and_display("Comprehensive SORA study report generated and saved as 'sora_study_report.txt'")

    def prepare_statistical_analysis_summary(self):
        try:
            summary = "T-test Results:\n"
            if hasattr(self, 't_test_results') and self.t_test_results:
                for result in self.t_test_results:
                    summary += f"- {result['comparison']}: t-statistic = {result['t_statistic']:.4f}, p-value = {result['p_value']:.4f}\n"
            else:
                summary += "No T-test results available.\n"

            if hasattr(self, 'anova_results') and self.anova_results:
                summary += f"\nANOVA Results:\nF-statistic = {self.anova_results['f_statistic']:.4f}, p-value = {self.anova_results['p_value']:.4f}\n"
            else:
                summary += "\nNo ANOVA results available.\n"

            if hasattr(self, 'tukey_results') and self.tukey_results:
                summary += "\nTukey's Test Results:\n"
                for result in self.tukey_results:
                    summary += f"- {result['comparison']}: diff = {result['diff']:.4f}, p-value = {result['p_value']:.4f}\n"
            else:
                summary += "\nNo Tukey's test results available.\n"
        except Exception as e:
            summary = f"Error in preparing statistical analysis summary: {str(e)}"
        
        return summary

    def prepare_cluster_analysis_summary(self):
        try:
            if not hasattr(self, 'cluster_info') or not self.cluster_info:
                return "Cluster analysis was not performed or did not yield results."

            summary = f"Number of clusters: {self.cluster_info.get('num_clusters', 'N/A')}\n"
            summary += f"Cluster sizes: {self.cluster_info.get('cluster_sizes', 'N/A')}\n\n"
            
            if 'cluster_characteristics' in self.cluster_info:
                for i, characteristics in enumerate(self.cluster_info['cluster_characteristics']):
                    summary += f"Cluster {i} characteristics:\n"
                    for key, value in characteristics.items():
                        summary += f"- {key}: {value:.4f}\n"
                    summary += "\n"
            else:
                summary += "Detailed cluster characteristics are not available.\n"
        except Exception as e:
            summary = f"Error in preparing cluster analysis summary: {str(e)}"
        
        return summary

    def prepare_ethical_implications_summary(self):
        try:
            summary = "Ethical Implication Scores by Agent Type:\n"
            if hasattr(self, 'ethical_scores') and self.ethical_scores:
                for agent_type, score in self.ethical_scores.items():
                    summary += f"- {agent_type}: {score:.4f}\n"
            else:
                summary += "No ethical scores available.\n"

            if hasattr(self, 'ethical_complexity_correlation'):
                summary += f"\nCorrelation between ethical implications and scenario complexity: {self.ethical_complexity_correlation:.4f}\n"
            else:
                summary += "\nCorrelation between ethical implications and scenario complexity: Not calculated\n"

            if hasattr(self, 'ethical_performance_correlation'):
                summary += f"Correlation between ethical implications and agent performance: {self.ethical_performance_correlation:.4f}\n"
            else:
                summary += "Correlation between ethical implications and agent performance: Not calculated\n"
        except Exception as e:
            summary = f"Error in preparing ethical implications summary: {str(e)}"
        
        return summary

    def prepare_nn_performance_summary(self):
        try:
            if not self.loss_history:
                return "No neural network training data available."

            summary = f"Final Loss: {self.loss_history[-1]:.4f}\n"
            summary += f"Loss Reduction: {(self.loss_history[0] - self.loss_history[-1]) / self.loss_history[0] * 100:.2f}%\n"
            
            # Estimate convergence epoch
            convergence_threshold = 0.001  # Example threshold, adjust as needed
            converged_epochs = [i for i in range(1, len(self.loss_history)) 
                                if abs(self.loss_history[i] - self.loss_history[i-1]) < convergence_threshold]
            convergence_epoch = converged_epochs[0] if converged_epochs else "Not reached"
            summary += f"Convergence Epoch: {convergence_epoch}\n"
        except Exception as e:
            summary = f"Error in preparing neural network performance summary: {str(e)}"
        
        return summary

    def get_roles_distribution(self):
        try:
            roles = [agent.role for agent in self.agents]
            role_counts = {role: roles.count(role) for role in set(roles)}
            return ", ".join(f"{role}: {count}" for role, count in role_counts.items())
        except Exception as e:
            return f"Error in getting roles distribution: {str(e)}"

    def get_autogen_insights(self):
        try:
            communication_increase = self.calculate_communication_increase()
            insights_prompt = f"""
            The communication among agents increased by {communication_increase:.2f}% during the SORA study.
            Analyze this increase and explain its significance in the context of agent collaboration and problem-solving efficiency.
            """
            # Utilisation de LangChain pour générer des insights basés sur les données réelles
            insight = self.llm(insights_prompt)
            return insight

        except Exception as e:
            self.log_and_display(f"Error in generating AutoGen insights: {str(e)}")
            return "Error generating insights."

    def calculate_communication_increase(self):
        try:
            baseline_data = self.interaction_data[self.interaction_data['epoch'] < self.epochs // 2]
            post_baseline_data = self.interaction_data[self.interaction_data['epoch'] >= self.epochs // 2]

            baseline_communication = baseline_data['communication_count'].mean()
            post_baseline_communication = post_baseline_data['communication_count'].mean()

            if baseline_communication > 0:
                communication_increase = ((post_baseline_communication - baseline_communication) / baseline_communication) * 100
            else:
                communication_increase = 0

            return communication_increase

        except Exception as e:
            self.log_and_display(f"Error in calculating communication increase: {str(e)}")
            return 0.0

    def get_langchain_metrics(self):
        try:
            total_response_time = self.langchain_metrics.get("total_response_time", 0)
            total_requests = self.langchain_metrics.get("total_requests", 1)  # Avoid division by zero
            successful_requests = self.langchain_metrics.get("successful_requests", 0)
            complex_tasks_completed = self.langchain_metrics.get("complex_tasks_completed", 0)

            # Calculer les métriques spécifiques
            average_response_time = total_response_time / total_requests
            query_success_rate = (successful_requests / total_requests) * 100
            complex_tasks_rate = (complex_tasks_completed / total_requests) * 100

            # Générer un rapport en utilisant LangChain
            metrics_prompt = f"""
            During the SORA study, LangChain processed {total_requests} requests.
            The average response time was {average_response_time:.2f} seconds.
            The query success rate was {query_success_rate:.2f}%, and {complex_tasks_rate:.2f}% of the tasks were complex tasks.
            Analyze these results and discuss the implications for the performance of LangChain in this context.
            """

            # Utilisation de LangChain pour générer un rapport basé sur les métriques
            metrics_report = self.llm(metrics_prompt)
            return metrics_report

        except Exception as e:
            self.log_and_display(f"Error in generating LangChain metrics: {str(e)}")
            return "Error generating metrics."

    def evaluate_hypothesis(self, metric):
        try:
            score = self.metrics.get(metric, 0)
            if score > 0.7:
                return "Strongly Supported"
            elif score > 0.5:
                return "Moderately Supported"
            else:
                return "Not Supported"
        except Exception as e:
            return f"Error in evaluating hypothesis: {str(e)}"

    def summarize_key_findings(self):
        self.log_and_display("Summarizing key findings using LangChain...")

        # Préparer les données à résumer
        findings_data = {
            "behavioral_diversity": self.metrics['behavioral_diversity'],
            "metacognition": self.metrics['metacognition'],
            "adaptability": self.metrics['adaptability'],
            "transparency": self.metrics['transparency'],
            "social_complexity": self.metrics['social_complexity'],
            "cluster_info": self.cluster_info,
            "ethical_scores": self.interaction_data['scenario_ethical_implications'].mean(),
            "complexity_handled": self.interaction_data['scenario_complexity'].mean(),
            "total_requests": self.langchain_metrics["total_requests"],
            "successful_requests": self.langchain_metrics["successful_requests"],
            "complex_tasks_completed": self.langchain_metrics["complex_tasks_completed"]
        }

        # Format the findings into a human-readable string
        findings_text = f"""
        - Behavioral Diversity: {findings_data['behavioral_diversity']:.2f}
        - Metacognition: {findings_data['metacognition']:.2f}
        - Adaptability: {findings_data['adaptability']:.2f}
        - Transparency: {findings_data['transparency']:.2f}
        - Social Complexity: {findings_data['social_complexity']:.2f}
        - Average Ethical Score: {findings_data['ethical_scores']:.2f}
        - Average Complexity Handled: {findings_data['complexity_handled']:.2f}
        - Total Requests: {findings_data['total_requests']}
        - Successful Requests: {findings_data['successful_requests']}
        - Complex Tasks Completed: {findings_data['complex_tasks_completed']}
        - Number of Clusters: {findings_data['cluster_info']['num_clusters']}
        """

        # Utilisation de LangChain pour générer un résumé
        try:
            summary_prompt = f"""
            You are an AI specialized in analyzing and summarizing complex data. Based on the following metrics and findings from an AI-driven multi-agent study, provide a concise summary highlighting the key insights and outcomes of the study:
            
            {findings_text}
            
            Summarize the findings with a focus on their significance, implications, and any notable patterns or trends observed.
            """

            summary_result = self.llm.predict(summary_prompt)

            self.log_and_display("Key Findings Summary:")
            self.log_and_display(summary_result.strip())
            
            return summary_result.strip()
            
        except Exception as e:
            self.log_and_display(f"Error in summarizing key findings: {str(e)}")
            return "Error in summarizing key findings."

    def suggest_future_research(self):
        self.log_and_display("Suggesting future research directions using LangChain...")

        # Préparer les données pour les suggestions de recherche
        research_data = {
            "behavioral_diversity": self.metrics['behavioral_diversity'],
            "metacognition": self.metrics['metacognition'],
            "adaptability": self.metrics['adaptability'],
            "transparency": self.metrics['transparency'],
            "social_complexity": self.metrics['social_complexity'],
            "cluster_info": self.cluster_info,
            "ethical_scores": self.interaction_data['scenario_ethical_implications'].mean(),
            "complexity_handled": self.interaction_data['scenario_complexity'].mean(),
            "total_requests": self.langchain_metrics["total_requests"],
            "successful_requests": self.langchain_metrics["successful_requests"],
            "complex_tasks_completed": self.langchain_metrics["complex_tasks_completed"]
        }

        # Format the research data into a human-readable string
        research_text = f"""
        - Behavioral Diversity: {research_data['behavioral_diversity']:.2f}
        - Metacognition: {research_data['metacognition']:.2f}
        - Adaptability: {research_data['adaptability']:.2f}
        - Transparency: {research_data['transparency']:.2f}
        - Social Complexity: {research_data['social_complexity']:.2f}
        - Average Ethical Score: {research_data['ethical_scores']:.2f}
        - Average Complexity Handled: {research_data['complexity_handled']:.2f}
        - Total Requests: {research_data['total_requests']}
        - Successful Requests: {research_data['successful_requests']}
        - Complex Tasks Completed: {research_data['complex_tasks_completed']}
        - Number of Clusters: {research_data['cluster_info']['num_clusters']}
        """

        # Utilisation de LangChain pour générer des suggestions de recherche futures
        try:
            research_prompt = f"""
            Based on the following results from the SORA study, suggest potential directions for future research that could build on these findings. Focus on areas that could improve the study outcomes, explore new dimensions of agent behavior, or address observed challenges:
            
            {research_text}
            
            Provide three to five specific and actionable research directions that could be pursued in future studies.
            """

            research_suggestions = self.llm.predict(research_prompt)

            self.log_and_display("Suggested Future Research Directions:")
            self.log_and_display(research_suggestions.strip())
            
            return research_suggestions.strip()

        except Exception as e:
            self.log_and_display(f"Error in suggesting future research directions: {str(e)}")
            return "Error in suggesting future research directions."

    def provide_additional_insights(self):
        self.log_and_display("Generating additional insights using LangChain...")

        # Préparer les données clés pour les insights
        insights_data = {
            "behavioral_diversity": self.metrics['behavioral_diversity'],
            "metacognition": self.metrics['metacognition'],
            "adaptability": self.metrics['adaptability'],
            "transparency": self.metrics['transparency'],
            "social_complexity": self.metrics['social_complexity'],
            "cluster_info": self.cluster_info,
            "ethical_scores": self.interaction_data['scenario_ethical_implications'].mean(),
            "complexity_handled": self.interaction_data['scenario_complexity'].mean(),
            "communication_count": self.interaction_data['communication_count'].sum(),
        }

        # Format the insights data into a human-readable string
        insights_text = f"""
        - Behavioral Diversity: {insights_data['behavioral_diversity']:.2f}
        - Metacognition: {insights_data['metacognition']:.2f}
        - Adaptability: {insights_data['adaptability']:.2f}
        - Transparency: {insights_data['transparency']:.2f}
        - Social Complexity: {insights_data['social_complexity']:.2f}
        - Average Ethical Score: {insights_data['ethical_scores']:.2f}
        - Average Complexity Handled: {insights_data['complexity_handled']:.2f}
        - Total Communication Count: {insights_data['communication_count']}
        - Number of Clusters: {insights_data['cluster_info']['num_clusters']}
        """

        # Utilisation de LangChain pour générer des insights supplémentaires
        try:
            insights_prompt = f"""
            Based on the following results from the SORA study, generate additional insights that could provide deeper understanding or uncover subtle patterns. Focus on areas that may not have been directly addressed in the key metrics but could reveal important findings:

            {insights_text}

            Provide three to five specific additional insights that could be valuable for understanding the results of the study.
            """

            additional_insights = self.llm.predict(insights_prompt)

            self.log_and_display("Additional Insights Generated:")
            self.log_and_display(additional_insights.strip())
            
            return additional_insights.strip()

        except Exception as e:
            self.log_and_display(f"Error in generating additional insights: {str(e)}")
            return "Error in generating additional insights."

    def run_study(self):
        try:
            self.log_and_display("Starting SORA study...")
            self.setup()
            self.run_simulation()
            self.calculate_metrics()
            self.perform_statistical_analysis()
            self.visualize_results()
            
            # Perform cluster analysis and use the results
            cluster_labels = self.cluster_analysis()
            self.analyze_cluster_results(cluster_labels)
            
            self.analyze_ethical_implications()
            self.generate_report()
            self.log_and_display("SORA study completed. Results, visualizations, and report generated.")
        finally:
            self.cleanup()

    def analyze_cluster_results(self, cluster_labels):
        self.log_and_display("Analyzing cluster results...")
        try:
            unique_clusters = np.unique(cluster_labels)
            cluster_sizes = [np.sum(cluster_labels == cluster) for cluster in unique_clusters]
            
            self.cluster_info = {
                "num_clusters": len(unique_clusters),
                "cluster_sizes": cluster_sizes,
                "cluster_characteristics": []
            }

            # Assuming we have agent performance data stored in self.agent_performance
            agent_performance = self.agent_performance  # This should be a pandas DataFrame

            for cluster in unique_clusters:
                cluster_mask = cluster_labels == cluster
                cluster_agents = agent_performance[cluster_mask]
                
                cluster_characteristics = {
                    "size": cluster_sizes[cluster],
                    "avg_performance": cluster_agents['performance'].mean(),
                    "avg_complexity_handled": cluster_agents['complexity_handled'].mean(),
                    "avg_ethical_score": cluster_agents['ethical_score'].mean(),
                    "dominant_role": cluster_agents['role'].mode().iloc[0]
                }
                
                self.cluster_info["cluster_characteristics"].append(cluster_characteristics)
                
                self.log_and_display(f"Cluster {cluster}:")
                self.log_and_display(f"  Size: {cluster_characteristics['size']} agents")
                self.log_and_display(f"  Avg Performance: {cluster_characteristics['avg_performance']:.2f}")
                self.log_and_display(f"  Avg Complexity Handled: {cluster_characteristics['avg_complexity_handled']:.2f}")
                self.log_and_display(f"  Avg Ethical Score: {cluster_characteristics['avg_ethical_score']:.2f}")
                self.log_and_display(f"  Dominant Role: {cluster_characteristics['dominant_role']}")

            # Visualize cluster distributions
            self.visualize_cluster_distributions(agent_performance, cluster_labels)

            # Perform statistical tests to compare clusters
            self.compare_clusters(agent_performance, cluster_labels)

        except Exception as e:
            self.log_and_display(f"Error in cluster analysis: {str(e)}")

    def visualize_cluster_distributions(self, agent_performance, cluster_labels):
        try:
            features = ['performance', 'complexity_handled', 'ethical_score']
            X = agent_performance[features]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)

            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis')
            plt.title('Cluster Distribution of Agents')
            plt.xlabel('First Principal Component')
            plt.ylabel('Second Principal Component')
            plt.colorbar(scatter, label='Cluster')

            for cluster in np.unique(cluster_labels):
                cluster_center = X_pca[cluster_labels == cluster].mean(axis=0)
                plt.annotate(f'Cluster {cluster}', cluster_center)

            filepath = os.path.join(self.log_directory, 'cluster_distribution.png')
            plt.savefig(filepath)
            plt.close()
            self.log_and_display("Cluster distribution visualization saved as 'cluster_distribution.png'")

        except Exception as e:
            self.log_and_display(f"Error in cluster visualization: {str(e)}")

    def compare_clusters(self, agent_performance, cluster_labels):
        try:
            from scipy import stats
            
            features = ['performance', 'complexity_handled', 'ethical_score']
            unique_clusters = np.unique(cluster_labels)
            
            for feature in features:
                self.log_and_display(f"\nComparing clusters based on {feature}:")
                f_statistic, p_value = stats.f_oneway(*[
                    agent_performance[cluster_labels == cluster][feature]
                    for cluster in unique_clusters
                ])
                self.log_and_display(f"ANOVA results: F-statistic = {f_statistic:.4f}, p-value = {p_value:.4f}")
                
                if p_value < 0.05:
                    self.log_and_display(f"Significant difference found in {feature} across clusters.")
                else:
                    self.log_and_display(f"No significant difference found in {feature} across clusters.")
        
        except Exception as e:
            self.log_and_display(f"Error in cluster comparison: {str(e)}")
if __name__ == "__main__":
    # study = SORAStudy(num_agents=20, num_scenarios=50, epochs=100)
    study = SORAStudy(num_agents=3, num_scenarios=3, epochs=2)
    study.run_study()
