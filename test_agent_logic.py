import unittest
from unittest.mock import MagicMock, patch
from agent_logic import get_study_agent, run_agent

class TestAgentLogic(unittest.TestCase):

    def test_get_study_agent(self):
        """Test that get_study_agent returns an Agent with correct configuration."""
        agent = get_study_agent(model_name="test-model")
        self.assertEqual(agent.name, "StudyNotesAgent")
        self.assertEqual(agent.model, "test-model")
        self.assertIn("PDF Summarization", agent.instructions)

    @patch("agent_logic.Runner")
    def test_run_agent(self, mock_runner):
        """Test that run_agent calls Runner.run and returns final output."""
        # Setup mock
        mock_result = MagicMock()
        mock_result.final_output = "Test Summary"
        mock_runner.run.return_value = mock_result
        
        agent = MagicMock()
        user_message = "Summarize this."
        
        # Execute
        output = run_agent(agent, user_message)
        
        # Verify
        mock_runner.run.assert_called_once_with(agent, input=user_message)
        self.assertEqual(output, "Test Summary")

if __name__ == '__main__':
    unittest.main()
