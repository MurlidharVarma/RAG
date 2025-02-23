package org.aipeel.ragai;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.shell.command.annotation.CommandScan;

@CommandScan
@SpringBootApplication
public class RagaiApplication {

	public static void main(String[] args) {
		SpringApplication.run(RagaiApplication.class, args);
	}

}
