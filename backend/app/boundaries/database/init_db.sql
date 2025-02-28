-- MySQL Workbench Forward Engineering

SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';

-- -----------------------------------------------------
-- Schema studd
-- -----------------------------------------------------

-- -----------------------------------------------------
-- Schema studd
-- -----------------------------------------------------
CREATE SCHEMA IF NOT EXISTS `studd` DEFAULT CHARACTER SET utf8 ;
USE `studd` ;

-- -----------------------------------------------------
-- Table `studd`.`User`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `studd`.`User` (
  `user_id` CHAR(36) NOT NULL,
  `user_name` VARCHAR(45) NOT NULL,
  `user_password` VARCHAR(45) NOT NULL,
  `permission` INT NOT NULL,
  PRIMARY KEY (`user_id`))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `studd`.`Model`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `studd`.`Model` (
  `model_id` CHAR(36) NOT NULL,
  `user_id` CHAR(36) NOT NULL,
  `model_name` VARCHAR(45) NULL,
  `data_dim` INT NOT NULL,
  `dataType` VARCHAR(100) NOT NULL,
  PRIMARY KEY (`model_id`),
  INDEX `user_id_idx` (`user_id` ASC) VISIBLE,
  CONSTRAINT `user_id`
    FOREIGN KEY (`user_id`)
    REFERENCES `studd`.`User` (`user_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `studd`.`Device`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `studd`.`Device` (
  `device_id` CHAR(36) NOT NULL,
  `user_id` CHAR(36) NOT NULL,
  `model_id` CHAR(36) NOT NULL,
  `device_name` VARCHAR(45) NOT NULL,
  `ip_address` VARCHAR(15) NOT NULL,
  `port` INT NOT NULL,
  PRIMARY KEY (`device_id`),
  INDEX `user_id_idx` (`user_id` ASC) VISIBLE,
  INDEX `model_id_idx` (`model_id` ASC) VISIBLE,
  CONSTRAINT `user_id`
    FOREIGN KEY (`user_id`)
    REFERENCES `studd`.`User` (`user_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `model_id`
    FOREIGN KEY (`model_id`)
    REFERENCES `studd`.`Model` (`model_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
