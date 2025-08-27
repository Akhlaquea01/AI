/**
 * Complete LangChain.js + MistralAI usage guide
 * Covers:
 *  - PromptTemplate and ChatPromptTemplate
 *  - All model call patterns (invoke, stream, batch, generate)
 *  - Parameters and options
 *  - Good practices with comments
 */

import { PromptTemplate, ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatMistralAI } from "@langchain/mistralai";
import { HumanMessage, SystemMessage, AIMessage } from "@langchain/core/messages";

/**
 * 1. Setup
 * Store your API key in an environment variable.
 * Windows (PowerShell): setx MISTRAL_API_KEY "your_api_key"
 * Linux/Mac: export MISTRAL_API_KEY="your_api_key"
 */
const MISTRAL_API_KEY = process.env.MISTRAL_API_KEY || "";

/**
 * 2. Initialize the model
 * Options available:
 *  - model: model name (e.g., mistral-tiny, mistral-medium, mistral-large-latest)
 *  - apiKey: your API key
 *  - temperature: randomness (0 = deterministic, higher = more creative)
 *  - maxTokens: limit output length (optional)
 */
const model = new ChatMistralAI({
    model: "mistral-large-latest",
    apiKey: MISTRAL_API_KEY,
    temperature: 0.3,
    // maxTokens: 256, // optional
});

/**
 * 3. Using PromptTemplate for single-shot prompts
 */
const countryPrompt = new PromptTemplate({
    template: "What is the capital of {country}?",
    inputVariables: ["country"],
});

const formatted = await countryPrompt.format({ country: "India" });
console.log("PromptTemplate Output:", formatted);

/**
 * 4. Basic invoke() with plain string input
 */
const res1 = await model.invoke("Write a haiku about programming.");
console.log("\nPlain String Response:", res1.content);

/**
 * 5. invoke() with messages (System + Human + AI)
 */
const messages = [
    new SystemMessage("You are a helpful assistant that translates English to Italian."),
    new HumanMessage("How are you today?"),
];
const res2 = await model.invoke(messages);
console.log("\nMessages Response:", res2.content);

/**
 * 6. Using ChatPromptTemplate for structured, multi-role prompts
 */
const translationTemplate = ChatPromptTemplate.fromMessages([
    ["system", "Translate the following text to {language}."],
    ["user", "{text}"],
]);

const chatInput = await translationTemplate.invoke({
    language: "Spanish",
    text: "Good morning, have a nice day!",
});
const res3 = await model.invoke(chatInput);
console.log("\nChatPromptTemplate Response:", res3.content);

/**
 * 7. Streaming output (token-by-token)
 */
console.log("\nStreaming Response:");
const stream = await model.stream(messages);
for await (const chunk of stream) {
    process.stdout.write(chunk.content); // print as it's generated
}
console.log("\n--- End of Stream ---");

/**
 * 8. Batch requests (multiple prompts in one call)
 */
const batchPrompts = [
    [new HumanMessage("Say hello in French")],
    [new HumanMessage("Say hello in German")],
];
const batchResults = await model.batch(batchPrompts);
console.log("\nBatch Results:");
batchResults.forEach((r, i) => console.log(`Prompt ${i + 1}:`, r.content));

/**
 * 9. generate() for advanced control
 * Allows multiple generations and richer metadata.
 */
const generateRes = await model.generate(batchPrompts);
console.log("\nGenerate() Output:");
generateRes.generations.forEach((g, i) =>
    console.log(`Prompt ${i + 1}:`, g[0].text)
);

/**
 * 10. Error Handling
 */
try {
    const badRes = await model.invoke([]);
    console.log(badRes);
} catch (err) {
    console.error("\nHandled Error:", err.message);
}

/**
 * 11. AIMessage usage
 * Build a conversation including previous AI responses.
 */
const conversation = [
    new SystemMessage("You are a math tutor."),
    new HumanMessage("Explain Pythagoras theorem."),
    new AIMessage("The Pythagorean theorem states..."),
    new HumanMessage("Give an example with numbers."),
];
const convRes = await model.invoke(conversation);
console.log("\nConversation Response:", convRes.content);
