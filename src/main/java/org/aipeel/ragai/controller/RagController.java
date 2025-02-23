package org.aipeel.ragai.controller;

import jakarta.annotation.PostConstruct;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.QuestionAnswerAdvisor;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.document.Document;
import org.springframework.ai.reader.tika.TikaDocumentReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

@RestController
public class RagController {

    ChatClient chatClient;
    ChatClient chatClient2;
    VectorStore vectorStore;

    @Value("classpath:docs/World2025.pdf")
    Resource pdfFile;

    @Value("classpath:prompts/assistance-prompt.st")
    Resource assistancePrompt;

    public RagController(ChatClient.Builder builder, VectorStore vectorStore){
        this.chatClient = builder.build();
        this.chatClient2 = builder.defaultAdvisors(new QuestionAnswerAdvisor(vectorStore)).build();
        this.vectorStore = vectorStore;
    }

    @GetMapping("/chat")
    public ResponseEntity<String> chat(@RequestParam("query") String query){
        // get inputs from vector database
        List<Document> vectorDoc = this.vectorStore.similaritySearch(SearchRequest.builder().query(query).topK(3).build());
        String document ="";
        if(vectorDoc!=null){
            document = vectorDoc.stream()
                                .map(Document::getFormattedContent)
                                .collect(Collectors.joining("\n"));
        }
        String prompt = getPromptForAssistance(query, document);
        System.out.println("PROMPT++++++++++++++++START");
        System.out.println(prompt);
        System.out.println("PROMPT++++++++++++++++END");
        return ResponseEntity.ok(Objects.requireNonNull(this.chatClient.prompt(prompt).call().chatResponse()).getResult().getOutput().getText());
    }

    @GetMapping("/chat2")
    public ResponseEntity<String> chat2(@RequestParam("query") String query){
        return ResponseEntity.ok(this.chatClient2.prompt().user(query).call().content());
    }

    @PostConstruct
    public void loadVectorStore(){
        List<Document> documents = this.vectorStore.similaritySearch("What is the title?");
        if(documents!=null && !documents.isEmpty()){
            return;
        }

        TikaDocumentReader tikaDocumentReader = new TikaDocumentReader(pdfFile);
        TokenTextSplitter textSplitter = TokenTextSplitter.builder().withMaxNumChunks(500000).withChunkSize(100).build();
        vectorStore.accept(textSplitter.apply(tikaDocumentReader.get()));
    }

    public String getPromptForAssistance (String input, String document){
        PromptTemplate promptTemplate = new PromptTemplate(assistancePrompt);
        Prompt prompt = promptTemplate.create(Map.of("input", input, "documents", document));
        return prompt.getContents();
    }
}
