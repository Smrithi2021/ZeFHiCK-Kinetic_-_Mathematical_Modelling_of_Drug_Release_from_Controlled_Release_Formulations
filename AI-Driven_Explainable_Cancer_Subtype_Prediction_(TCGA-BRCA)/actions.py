from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

class ActionGenericResponse(Action):
    def name(self):
        return "action_generic_response"

    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message(
            text="I can help explain breast cancer subtypes and AI predictions."
        )
        return []
