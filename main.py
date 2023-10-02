import logging
from fastapi import FastAPI

from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.synthesizer import AzureSynthesizerConfig
from vocode.streaming.synthesizer.azure_synthesizer import AzureSynthesizer
from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig
from vocode.streaming.synthesizer.eleven_labs_synthesizer import ElevenLabsSynthesizer
from vocode.streaming.models.synthesizer import PlayHtSynthesizerConfig
from vocode.streaming.synthesizer.play_ht_synthesizer import PlayHtSynthesizer


from vocode.streaming.agent.chat_gpt_agent import ChatGPTAgent
from vocode.streaming.client_backend.conversation import ConversationRouter
from vocode.streaming.models.message import BaseMessage

from dotenv import load_dotenv

load_dotenv()

app = FastAPI(docs_url=None)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

text = "I want you to act as a mental health adviser. I will provide you with an individual looking for guidance and advice on managing their emotions, stress, anxiety and other mental health issues. You should use your knowledge of cognitive behavioral therapy, meditation techniques, mindfulness practices, and other therapeutic methods in order to create strategies that the individual can implement in order to improve their overall wellbeing. Please keep your replies short."

conversation_router = ConversationRouter(
    agent_thunk=lambda: ChatGPTAgent(
        ChatGPTAgentConfig(
            initial_message=BaseMessage(text="Hi there!"),
            prompt_preamble=text,
        )
    ),
    synthesizer_thunk=lambda output_audio_config: ElevenLabsSynthesizer(
         ElevenLabsSynthesizerConfig.from_output_audio_config(
            output_audio_config, voice_id="pMsXgVXv3BLzUgSXRplE"
        )
    ),
    logger=logger,
)

app.include_router(conversation_router.get_router())
