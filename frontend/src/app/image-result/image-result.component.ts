import { Component, Input, OnInit, Output, EventEmitter  } from '@angular/core';


@Component({
  selector: 'app-image-result',
  templateUrl: './image-result.component.html',
  styleUrls: ['./image-result.component.css']
})
export class ImageResultComponent implements OnInit {

  @Input() imgUrl: any;
  @Input() detectedEmotion: any;
  @Output() backEvent = new EventEmitter();
  public results = {
    valid: 0,
    invalid: 0
  };
  isVoted = false;

  constructor() { }

  ngOnInit(): void {
    const validResults = localStorage.getItem('validResults');
    const invalidResults = localStorage.getItem('invalidResults');
    if (validResults !== null && invalidResults !== null){
      this.results = {
        valid: +validResults,
        invalid: +invalidResults
      };
    }
  }

  setUserResult(value: boolean): void{
    if (value){
      this.results.valid += 1;
      localStorage.setItem('validResults', this.results.valid.toString());
    } else {
      this.results.invalid += 1;
      localStorage.setItem('invalidResults', this.results.invalid.toString());
    }
    this.isVoted = true;
  }

  onBack(): void{
    this.backEvent.emit(true);
  }

}
